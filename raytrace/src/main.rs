
use raytrace_lib::raytrace::{make_vec, make_color, make_disk, RayCaster,
                             DefaultRayCaster, Scene, SurfaceKind,
                             Triangle, Viewport};
use raytrace_lib::obj_parser;
use raytrace_lib::raytrace;

use std::time;
use std::fs;
use std::io::Result;
use std::path::Path;
use std::collections::HashMap;

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

    let aspect = 480. / 640.;
    let width = 640;
    let height = 480;

    // let aspect = 480. / 640.;
    // let width = 160;
    // let height = 120;

    // let aspect = 1. / 1.;
    // let width = 1;
    // let height = 1;

    let mut obj_data: Vec<Triangle> = Vec::new();
    obj_data.extend(obj_parser::parse_obj("teapot_tri.obj",
                                         &make_vec(&[0., 0.5, 5.]),
                                         1.0,
                                         raytrace::create_transform(&make_vec(&[0., 0.3, 1.]).unit(),
                                                                    270_f32.to_radians()),
                                        //  &SurfaceKind::Solid { color: make_color((252, 119, 0))},
                                         &SurfaceKind::Matte { color: make_color((252, 119, 0)),
                                                               alpha: 0.2 },
                                         0.05));

    let v = raytrace::create_viewport((width, height),
                            (1., 1. * aspect),
                            &make_vec(&[2., 0., 0.]),
                            &make_vec(&[0., 0., 1.]).unit(),
                            90.,
                            0_f32.to_radians(),
                            5,
                            10);

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

    // optimize(&obj_data, &v, (10, 15));
    // return Ok(());

    // let bbox = raytrace::build_trivial_bounding_box(&obj_data,
    //                                         &make_vec(&[0., 0., 0.]),
    //                                         20.);
    let bbox = raytrace::build_bounding_box(&obj_data,
                                            &make_vec(&[0., 0., 20.1]),
                                            20.,
                                            7,
                                            19);


    let s = Scene {
        tris: obj_data,
        boxes: bbox,
    };


    let caster = DefaultRayCaster {};

    // let mut data = vec![make_vec(&[0., 0., 0.]); 1 as usize ];
    // let progress_ctx = v.walk_one_ray(&s, &mut data, (416, 130), &caster);
    // let _ = raytrace::write_png(file, (1, 1), &data);

    let mut data = vec![make_vec(&[0., 0., 0.]); (width*height) as usize ];
    let progress_ctx = caster.walk_rays(&v, &s, &mut data, 1, true);
    progress_ctx.print_stats();
    let _ = raytrace::write_png(file, (width, height), &data);

    return Ok(());
}
