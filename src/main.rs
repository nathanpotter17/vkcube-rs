use sdl2::{event::Event, EventPump};
use std::sync::Arc;

use simmerlib::{
    device::{DeviceContext, WINDOW_NAME},
    memory::{MemoryContext, UnifiedMemoryPool, WIDTH, HEIGHT, WORKER_THREADS, ENABLE_VALIDATION},
    renderer::Renderer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════╗");
    println!("║        VULKAN RENDERER             ║");
    println!("║   Beta II Memory Architecture      ║");
    println!("╚════════════════════════════════════╝");
    
    // Initialize SDL2
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    
    // Create window
    let window = video_subsystem
        .window(WINDOW_NAME, WIDTH, HEIGHT)
        .position_centered()
        .resizable()
        .vulkan()
        .build()?;
    
    // Initialize Vulkan
    let mut device_context = DeviceContext::new(&window, ENABLE_VALIDATION)?;
    
    // Initialize Beta II memory system
    let memory_pool = Arc::new(UnifiedMemoryPool::new(
        device_context.device.clone(),
        device_context.memory_properties,
        device_context.command_pool,
        device_context.queue,
        device_context.queue_family_index,
        WORKER_THREADS,
    )?);
    
    let memory_ctx = MemoryContext::from_pool(memory_pool);
    
    println!("✓ Device initialized");
    println!("✓ Memory system initialized");
    
    // Create renderer
    let mut renderer = Renderer::new(&device_context, memory_ctx.clone())?;
    println!("✓ Renderer initialized");
    
    // Create event pump
    let mut event_pump: EventPump = sdl_context.event_pump()?;
    println!("\n✓ SDL Events Active...\n");
    
    // Main loop
    println!("\nStarting render loop...\n");
    'running: loop {
        // Handle events
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } |
                Event::KeyDown { keycode: Some(sdl2::keyboard::Keycode::Escape), .. } => {
                    break 'running;
                }
                Event::Window { win_event: sdl2::event::WindowEvent::Resized(w, h), .. } => {
                    println!("Window resized: {}x{}", w, h);
                    device_context.recreate_swapchain(w as u32, h as u32);
                    renderer.recreate_framebuffers(&device_context)?;
                }
                _ => {}
            }
        }
        
        // Render frame
        renderer.render(&device_context)?;
    }
    
    // Cleanup happens via Drop implementations
    println!("\nShutting down...");
    Ok(())
}