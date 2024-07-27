
use raytrace_lib::debug::make_debug_ctx;
use raytrace_lib::raytrace::{make_color, make_disk, make_dummy_triangle, make_vec, populate_triangle_numbers, DefaultRayCaster, RayCaster, Scene, SurfaceKind, Triangle, Viewport};
use raytrace_lib::obj_parser;
use raytrace_lib::raytrace;

use cuda_raytrace_lib::cuda_raytrace;

use std::time;
use std::fs;
use std::io::Result;
use std::path::Path;
use std::collections::HashMap;

use sdl2;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;

#[allow(dead_code)]
fn optimize(tris: &Vec<Triangle>, v: &Viewport, initial: (usize, usize)) {

    let mut max_box_depth = initial.0;
    let mut objs_per_box = initial.1;

    let mut runtimes: HashMap<(usize, usize), time::Duration> = HashMap::new();

    loop {

        let mut min_time = u64::MAX;
        let mut min_vals = (0, 0);
        let mut min_idx = 0;
        for idx in 0..9 {
            let x = ((idx % 3) as isize - 1) * 1;
            let y = ((idx / 3) as isize - 1) * 1;
            let run_vals = ((max_box_depth as isize + x) as usize,
                            (objs_per_box as isize + y) as usize);
            if !runtimes.contains_key(&run_vals) {
                println!("Running iteration {} {}", run_vals.0, run_vals.1);
                let runtime = run_iteration(tris, v, run_vals);
                runtimes.insert(run_vals, runtime);
                println!("Runtime: {:.3}", runtime.as_secs_f64());
            }

            let time = runtimes.get(&run_vals).unwrap().as_micros() as u64;

            if time < min_time {
                min_vals = run_vals;
                min_idx = idx;
                min_time = time;
            }
        }

        if min_idx == 4 {
            break;
        } else {
            max_box_depth = min_vals.0;
            objs_per_box = min_vals.1;
        }
    }

    println!("Found minimum at {} {}", max_box_depth, objs_per_box)
}

#[allow(dead_code)]
fn run_iteration(tris: &Vec<Triangle>, v: &Viewport, initial: (usize, usize)) -> time::Duration {
    let bbox = raytrace::build_bounding_box(&tris,
                                            &make_vec(&[0., 0., 0.]),
                                            20.,
                                            initial.0,
                                            initial.1);

    let s = Scene {
        tris: tris.clone(),
        boxes: bbox,
        debug_ctx: make_debug_ctx().into(),
        debug_en: false
    };

    let caster = DefaultRayCaster {};

    let mut data = vec![make_vec(&[0., 0., 0.]); (v.width*v.height) as usize ];
    let t1 = time::Instant::now();
    caster.walk_rays(&v, &s, &mut data, 16, false);
    let t2 = time::Instant::now();

    t2-t1
}

fn main() -> Result<()> {

    env_logger::init();

    let path = Path::new("test.png");
    let file = fs::File::create(&path)?;

    // let aspect = 2160. / 3840.;
    // let width = 3840;
    // let height = 2160;

    // let aspect = 1440. / 2560.;
    // let width = 2560;
    // let height = 1440;

    // let aspect = 480. / 640.;
    // let width = 640;
    // let height = 480;

    let aspect = 64. / 64.;
    let width = 64;
    let height = 64;

    // let aspect = 1. / 1.;
    // let width = 1;
    // let height = 1;

    let mut obj_data: Vec<Triangle> = Vec::new();
    obj_data.push(make_dummy_triangle());
    obj_data.extend(obj_parser::parse_obj("teapot_tri.obj",
                                         &make_vec(&[0., 0.5, 5.]),
                                         1.0,
                                         raytrace::create_transform(&make_vec(&[0., 0.3, 1.]).unit(),
                                                                    270_f32.to_radians()),
                                        //  &SurfaceKind::Solid { color: make_color((252, 119, 0))},
                                         &SurfaceKind::Matte { color: make_color((252, 119, 0)),
                                                               alpha: 0.2 },
                                         0.05));

    obj_data.extend(make_disk(&make_vec(&[4., 4., 7.]),
                              &make_vec(&[-0.3, -0.55, -0.5]).unit(),
                              2.,
                              0.1,
                              50,
                              &SurfaceKind::Reflective { scattering: 0.0002,
                                           color: make_color((230, 230, 230)),
                                           alpha: 0.7 },
                              &SurfaceKind::Matte { color: make_color((40, 40, 40)),
                                           alpha: 0.2 },
                              -1.));

    obj_data.extend(make_disk(&make_vec(&[4., -3., 5.]),
                              &make_vec(&[-0.5, 2.0, -0.5]).unit(),
                              1.,
                              0.04,
                              50,
                              &SurfaceKind::Reflective { scattering: 0.002,
                                           color: make_color((230, 230, 230)),
                                           alpha: 0.7 },
                              &SurfaceKind::Matte { color: make_color((40, 40, 40)),
                                           alpha: 0.2 },
                              -1.));

    populate_triangle_numbers(&mut obj_data);

    // optimize(&obj_data, &v, (10, 15));
    // return Ok(());

    // let bbox = raytrace::build_trivial_bounding_box(&obj_data,
    //                                         &make_vec(&[0., 0., 0.]),
    //                                         20.);
    let bbox = raytrace::build_bounding_box(&obj_data,
                                            &make_vec(&[0., 0., 20.1]),
                                            20.,
                                            10,
                                            19);

    let v = raytrace::create_viewport((width, height),
                            (1., 1. * aspect),
                            &make_vec(&[2., 0., 0.]),
                            &make_vec(&[0., 0., 1.]).unit(),
                            90.,
                            0_f32.to_radians(),
                            5,
                            1);


    let s_default = Scene {
        tris: obj_data,
        boxes: bbox,
        debug_ctx: make_debug_ctx().into(),
        debug_en: true
    };

    let caster_default = DefaultRayCaster {};
    let caster_cuda = cuda_raytrace::CudaRayCaster {};

    // let mut data = vec![make_vec(&[0., 0., 0.]); 1 as usize ];
    // let progress_ctx = v.walk_one_ray(&s, &mut data, (416, 130), &caster);
    // let _ = raytrace::write_png(file, (1, 1), &data);

    let mut data = vec![make_vec(&[0., 0., 0.]); (width*height) as usize ];
    let progress_ctx = caster_default.walk_rays(&v, &s_default, &mut data, 1, false);

    let s_cuda = Scene {
        tris: s_default.tris,
        boxes: s_default.boxes,
        debug_ctx: make_debug_ctx().into(),
        debug_en: true
    };

    let progress_ctx = caster_cuda.walk_rays(&v, &s_cuda, &mut data, 1, false);
    // return Ok(());

    progress_ctx.print_stats();
    let _ = raytrace::write_png(file, (width, height), &data);

    if s_default.debug_en {
        let mut debug_f = fs::File::create("debug_default.csv").unwrap();
        let debug_ctx = s_default.debug_ctx.lock().unwrap();
        debug_ctx.write_debug_header(&mut debug_f);
        debug_ctx.write_all_debug_context(&mut debug_f);
    }

    if s_cuda.debug_en {
        let mut debug_f = fs::File::create("debug_cuda.csv").unwrap();
        let debug_ctx = s_cuda.debug_ctx.lock().unwrap();
        debug_ctx.write_debug_header(&mut debug_f);
        debug_ctx.write_all_debug_context(&mut debug_f);
    }

    if s_default.debug_en && s_cuda.debug_en {

        let mut debug_f = fs::File::create("debug_diffs.txt").unwrap();
        let debug_ctx_default = s_default.debug_ctx.lock().unwrap();
        let debug_ctx_cuda = s_cuda.debug_ctx.lock().unwrap();

        debug_ctx_default.compare_to(&debug_ctx_cuda, &mut debug_f);
    }

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem.window("Results", width, height)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();
    canvas.set_draw_color(sdl2::pixels::Color::RGB(0, 0, 0));
    canvas.clear();

    let texture_creator = canvas.texture_creator();
    let mut texture = texture_creator.create_texture_static(sdl2::pixels::PixelFormatEnum::BGR888, width, height).unwrap();

    let mut data_u8 = vec![0; (width*height*4) as usize];

    for (i, item) in data.iter().enumerate() {
        data_u8[i*4] = (item.v[0] * 255.) as u8;
        data_u8[i*4 + 1] = (item.v[1] * 255.) as u8;
        data_u8[i*4 + 2] = (item.v[2] * 255.) as u8;
        data_u8[i*4 + 3] = 255;
    }

    texture.update(sdl2::rect::Rect::new(0, 0, width, height), data_u8.as_slice(), (width * 4) as usize).unwrap();

    canvas.copy(&texture, None, None).unwrap();

    canvas.present();

    let mut event_pump = sdl_context.event_pump().unwrap();
    'run_loop: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} |
                Event::KeyDown { keycode: Some(Keycode::Escape), .. } => {
                    break 'run_loop
                },
                _ => {}
            }
        }
    };

    Ok(())
}
