use std::{
    time::{Duration, Instant},
    collections::VecDeque,
    sync::{Arc, Mutex, mpsc},
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};

// ===== Core Event Types =====

#[derive(Debug, Clone)]
pub enum EngineEvent {
    RequestRedraw,
    ResourceLoaded { id: usize, name: String },
    ExecuteCommand(Command),
    Shutdown,
}

#[derive(Debug, Clone)]
pub enum Command {
    ReloadShaders,
    TakeScreenshot,
    Custom { name: String, data: Vec<u8> },
}

// ===== Priority =====

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    High = 0,    // User input, rendering, critical
    Normal = 1,  // Resource loading, general commands  
    Low = 2,     // Background tasks, maintenance
}

impl Default for Priority {
    fn default() -> Self { Priority::Normal }
}

// ===== Batched Event Queue =====

struct EventQueue {
    high: Mutex<VecDeque<EngineEvent>>,
    normal: Mutex<VecDeque<EngineEvent>>,
    low: Mutex<VecDeque<EngineEvent>>,
    batch_size: usize,
    last_flush: Mutex<Instant>,
    flush_interval: Duration,
}

impl EventQueue {
    fn new(batch_size: usize) -> Self {
        Self {
            high: Mutex::new(VecDeque::new()),
            normal: Mutex::new(VecDeque::new()),
            low: Mutex::new(VecDeque::new()),
            batch_size,
            last_flush: Mutex::new(Instant::now()),
            flush_interval: Duration::from_millis(16), // ~60fps
        }
    }
    
    fn push(&self, event: EngineEvent, priority: Priority) {
        let queue = match priority {
            Priority::High => &self.high,
            Priority::Normal => &self.normal,
            Priority::Low => &self.low,
        };
        
        if let Ok(mut q) = queue.lock() {
            q.push_back(event);
        }
    }
    
    fn should_flush(&self) -> bool {
        let now = Instant::now();
        let last = *self.last_flush.lock().unwrap();
        
        // Flush if interval passed or high priority queue has items
        now.duration_since(last) >= self.flush_interval ||
        self.high.lock().unwrap().len() > 0
    }
    
    fn drain_batch(&self) -> Vec<EngineEvent> {
        let mut batch = Vec::with_capacity(self.batch_size);
        
        // Priority order: drain high first, then normal, then low
        for queue in [&self.high, &self.normal, &self.low] {
            if let Ok(mut q) = queue.lock() {
                while batch.len() < self.batch_size && !q.is_empty() {
                    if let Some(event) = q.pop_front() {
                        batch.push(event);
                    }
                }
            }
        }
        
        if !batch.is_empty() {
            *self.last_flush.lock().unwrap() = Instant::now();
        }
        
        batch
    }
}

// ===== Simple Rate Limiter =====

struct RateLimiter {
    window_start: Mutex<Instant>,
    event_count: AtomicUsize,
    max_per_second: usize,
    dropped: AtomicU64,
}

impl RateLimiter {
    fn new(max_per_second: usize) -> Self {
        Self {
            window_start: Mutex::new(Instant::now()),
            event_count: AtomicUsize::new(0),
            max_per_second,
            dropped: AtomicU64::new(0),
        }
    }
    
    fn check(&self) -> bool {
        let now = Instant::now();
        let mut window = self.window_start.lock().unwrap();
        
        // Reset window every second
        if now.duration_since(*window) >= Duration::from_secs(1) {
            *window = now;
            self.event_count.store(0, Ordering::Release);
        }
        
        let count = self.event_count.fetch_add(1, Ordering::AcqRel);
        if count < self.max_per_second {
            true
        } else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            false
        }
    }
}

// ===== SDL2-Compatible Event Proxy =====

#[derive(Clone)]
pub struct EventProxy {
    sender: mpsc::Sender<EngineEvent>,
    queue: Arc<EventQueue>,
    limiter: Arc<RateLimiter>,
    flush_handle: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
}

impl EventProxy {
    pub fn new() -> (Self, mpsc::Receiver<EngineEvent>) {
        let (tx, rx) = mpsc::channel();
        let sender_clone = tx.clone();
        let queue = Arc::new(EventQueue::new(32));
        let limiter = Arc::new(RateLimiter::new(1000));
        
        let result = Self {
            sender: tx,
            queue: queue.clone(),
            limiter,
            flush_handle: Arc::new(Mutex::new(None)),
        };
        
        // Start background flusher
        let queue_clone = queue.clone();
        let handle = std::thread::spawn(move || {
            loop {
                std::thread::sleep(Duration::from_millis(8));
                
                if queue_clone.should_flush() {
                    let batch = queue_clone.drain_batch();
                    for event in batch {
                        if sender_clone.send(event).is_err() {
                            // Receiver dropped, exit thread
                            break;
                        }
                    }
                }
            }
        });
        
        *result.flush_handle.lock().unwrap() = Some(handle);
        (result, rx)
    }
    
    // Send with automatic priority detection
    pub fn send(&self, event: EngineEvent) -> bool {
        let priority = match &event {
            EngineEvent::Shutdown => {
                // Shutdown bypasses queue
                return self.sender.send(event).is_ok();
            }
            EngineEvent::RequestRedraw => Priority::High,
            EngineEvent::ExecuteCommand(Command::TakeScreenshot) => Priority::High,
            EngineEvent::ExecuteCommand(_) => Priority::Normal,
            EngineEvent::ResourceLoaded { .. } => Priority::Low,
        };
        
        self.send_with_priority(event, priority)
    }
    
    pub fn send_with_priority(&self, event: EngineEvent, priority: Priority) -> bool {
        if !self.limiter.check() {
            return false;
        }
        
        self.queue.push(event, priority);
        true
    }
    
    // Convenience methods
    pub fn request_redraw(&self) {
        self.send(EngineEvent::RequestRedraw);
    }
    
    pub fn notify_resource_loaded(&self, id: usize, name: impl Into<String>) {
        self.send(EngineEvent::ResourceLoaded { 
            id, 
            name: name.into() 
        });
    }
    
    pub fn execute_command(&self, cmd: Command) {
        self.send(EngineEvent::ExecuteCommand(cmd));
    }
    
    pub fn shutdown(&self) {
        self.send(EngineEvent::Shutdown);
    }
    
    // Spawn task with event proxy access
    pub fn spawn_task<F>(&self, task: F)
    where
        F: FnOnce(EventProxy) + Send + 'static,
    {
        let proxy = self.clone();
        std::thread::spawn(move || task(proxy));
    }
    
    // Force immediate flush (useful for testing)
    pub fn flush(&self) {
        let batch = self.queue.drain_batch();
        for event in batch {
            let _ = self.sender.send(event);
        }
    }
    
    // Get stats
    pub fn dropped_events(&self) -> u64 {
        self.limiter.dropped.load(Ordering::Relaxed)
    }
}

impl Default for EventProxy {
    fn default() -> Self {
        let (proxy, _rx) = Self::new();
        proxy
    }
}