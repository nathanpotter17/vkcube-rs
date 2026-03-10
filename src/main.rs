use std::time::Instant;

use sdl2::event::Event;
use sdl2::keyboard::Scancode;
use sdl2::mouse::MouseButton;

use simmerlib::{
    device::{DeviceContext, WINDOW_NAME},
    memory::{MemoryContext, WIDTH, HEIGHT, ENABLE_VALIDATION},
    renderer::Renderer,
    scene::{InputAction, InputState},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n╔════════════════════════════════════╗");
    println!("║        VULKAN RENDERER             ║");
    println!("║   VMA-Style Memory Architecture    ║");
    println!("║   + Async Streaming & Budgeting    ║");
    println!("╠════════════════════════════════════╣");
    println!("║  Controls (UE5 fly mode):          ║");
    println!("║    RMB + Mouse  = Look around      ║");
    println!("║    W/A/S/D      = Move              ║");
    println!("║    Q / E        = Down / Up         ║");
    println!("║    Shift        = Move fast         ║");
    println!("║    Scroll       = Adjust speed      ║");
    println!("║    L            = Spawn point light ║");
    println!("║    G            = Spawn geometry    ║");
    println!("║    Escape       = Quit              ║");
    println!("╚════════════════════════════════════╝");

    // SDL2
    let sdl_context = sdl2::init()?;
    let video = sdl_context.video()?;
    let mouse_util = sdl_context.mouse();

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
    let mut event_pump = sdl_context.event_pump()?;
    println!("\nStarting render loop...\n");

    let mut last_frame = Instant::now();
    let mut rmb_held = false;

    'running: loop {
        // ---- Accumulate per-frame input from SDL2 events ----
        let mut mouse_dx: i32 = 0;
        let mut mouse_dy: i32 = 0;
        let mut scroll_y: f32 = 0.0;
        let mut actions: Vec<InputAction> = Vec::new();

        for event in event_pump.poll_iter() {
            match event {
                // ---- Quit ----
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(sdl2::keyboard::Keycode::Escape),
                    ..
                } => break 'running,

                // ---- Window resize ----
                Event::Window {
                    win_event: sdl2::event::WindowEvent::Resized(w, h),
                    ..
                } => {
                    println!("Window resized: {}x{}", w, h);
                    device_context.recreate_swapchain(w as u32, h as u32);
                    renderer.recreate_framebuffers(&device_context)?;
                }

                // ---- Mouse motion (accumulated for the frame) ----
                Event::MouseMotion { xrel, yrel, .. } => {
                    mouse_dx += xrel;
                    mouse_dy += yrel;
                }

                // ---- Right mouse button: toggle look mode ----
                Event::MouseButtonDown { mouse_btn: MouseButton::Right, .. } => {
                    rmb_held = true;
                    mouse_util.set_relative_mouse_mode(true);
                }
                Event::MouseButtonUp { mouse_btn: MouseButton::Right, .. } => {
                    rmb_held = false;
                    mouse_util.set_relative_mouse_mode(false);
                }

                // ---- Scroll wheel: speed adjustment ----
                Event::MouseWheel { y, .. } => {
                    scroll_y += y as f32;
                }

                // ---- Single-press key actions (non-repeat) ----
                Event::KeyDown { keycode: Some(kc), repeat: false, .. } => {
                    match kc {
                        sdl2::keyboard::Keycode::L => actions.push(InputAction::SpawnLight),
                        sdl2::keyboard::Keycode::G => actions.push(InputAction::SpawnGeometry),
                        _ => {}
                    }
                }

                _ => {}
            }
        }

        // ---- Build continuous key state from SDL2 keyboard snapshot ----
        let ks = event_pump.keyboard_state();
        let input = InputState {
            move_forward: ks.is_scancode_pressed(Scancode::W),
            move_back:    ks.is_scancode_pressed(Scancode::S),
            move_left:    ks.is_scancode_pressed(Scancode::A),
            move_right:   ks.is_scancode_pressed(Scancode::D),
            move_up:      ks.is_scancode_pressed(Scancode::E),
            move_down:    ks.is_scancode_pressed(Scancode::Q),
            fast:         ks.is_scancode_pressed(Scancode::LShift)
                       || ks.is_scancode_pressed(Scancode::RShift),
            mouse_look: rmb_held,
            mouse_dx: mouse_dx as f32,
            mouse_dy: mouse_dy as f32,
            scroll_y,
            actions,
        };

        // ---- Delta time ----
        let now = Instant::now();
        let dt = now.duration_since(last_frame).as_secs_f32().min(0.1);
        last_frame = now;

        renderer.render(&device_context, &input, dt)?;
    }

    // Drop order: renderer (+ memory_ctx inside it) → device_context
    println!("\nShutting down...");
    Ok(())
}