use sdl2::{event::Event, EventPump};

use simmerlib::{
    device::{DeviceContext, WINDOW_NAME},
    memory::{MemoryContext, WIDTH, HEIGHT, ENABLE_VALIDATION},
    renderer::Renderer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════╗");
    println!("║        VULKAN RENDERER             ║");
    println!("║   VMA-Style Memory Architecture    ║");
    println!("║   + Async Streaming & Budgeting    ║");
    println!("╚════════════════════════════════════╝");

    // SDL2
    let sdl_context = sdl2::init()?;
    let video = sdl_context.video()?;

    let window = video
        .window(WINDOW_NAME, WIDTH, HEIGHT)
        .position_centered()
        .resizable()
        .vulkan()
        .build()?;

    // Vulkan device (now discovers dedicated transfer queue)
    let mut device_context = DeviceContext::new(&window, ENABLE_VALIDATION)?;
    println!(
        "✓ Device initialized (UBO alignment: {}, dedicated transfer: {})",
        device_context.min_ubo_alignment,
        device_context.has_dedicated_transfer,
    );

    // Memory system: pool allocator + ring buffer + transfer queue + budget
    let memory_ctx = MemoryContext::new(
        device_context.device.clone(),
        device_context.memory_properties,
        device_context.min_ubo_alignment,
        device_context.transfer_queue,
        device_context.transfer_queue_family_index,
        device_context.queue_family_index,
    )?;
    println!("✓ Memory system initialized (with async transfer + budget)");

    // Renderer (takes ownership of MemoryContext)
    let mut renderer = Renderer::new(&device_context, memory_ctx)?;
    println!("✓ Renderer initialized");

    // Event loop
    let mut event_pump: EventPump = sdl_context.event_pump()?;
    println!("\nStarting render loop...\n");

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(sdl2::keyboard::Keycode::Escape),
                    ..
                } => break 'running,

                Event::Window {
                    win_event: sdl2::event::WindowEvent::Resized(w, h),
                    ..
                } => {
                    println!("Window resized: {}x{}", w, h);
                    device_context.recreate_swapchain(w as u32, h as u32);
                    renderer.recreate_framebuffers(&device_context)?;
                }

                _ => {}
            }
        }

        renderer.render(&device_context)?;
    }

    // Drop order: renderer (+ memory_ctx inside it) → device_context
    println!("\nShutting down...");
    Ok(())
}