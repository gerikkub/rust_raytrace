
use raytrace_lib::progress::ProgressStat;
use raytrace_lib::raytrace::{make_vec, make_color, make_disk, make_sphere, make_triangle,
                             BoundingBox, CollisionFace, Color, DefaultRayCaster, LightSource,
                             Point, Ray, Scene, SurfaceKind, Triangle, Vec3, Viewport};
use raytrace_lib::obj_parser;
use raytrace_lib::raytrace;
use raytrace_lib::raytrace::RayCaster;

use std::time;
use std::fs;
use std::io::Result;
use std::path::Path;
use std::collections::HashMap;

use ffi::CudaTriangle;

#[cxx::bridge]
mod ffi {

    struct CudaVec3 {
        v: [f32; 3]
    }

    struct CudaRay {
        a: CudaVec3,
        u: CudaVec3
    }

    struct CudaColor {
        c: [u8; 3]
    }

    struct CudaTriangle {
        incenter: CudaVec3,
        norm: CudaVec3,
        sides: [CudaVec3; 3],
        side_lens: [f32; 3]
    }


    extern "Rust" {

        type CudaRayCaster;

    }

    unsafe extern "C++" {
        include!("cuda_raytrace/src/cuda_rt.h");

        fn preload_triangles_cuda(tris: &Vec<CudaTriangle>);

        fn project_ray_cuda<'a>(r: &CudaRay, objs: &Vec<usize>, ignore_objid: usize,
                                depth: usize, runtimes: &mut [u64; 3]) -> CudaColor;
    }

}

struct CudaRayCaster {
}


// impl From<Vec3> for ffi::CudaVec3 {
//     fn from(item: Vec3) -> Self {
//         ffi::CudaVec3 {v: [item.0 as f32, item.1 as f32, item.2 as f32]}
//     }
// }

// impl From<Ray> for ffi::CudaRay {
//     fn from(item: Ray) -> Self {
//         ffi::CudaRay {
//             a: item.orig.into(),
//             u: item.dir.into()
//         }
//     }
// }

// impl Into<Color> for ffi::CudaColor {
//     fn into(self) -> Color {
//         make_color((self.c[0], self.c[1], self.c[2]))
//     }
// }

// impl From<Triangle> for ffi::CudaTriangle {
//     fn from(item: Triangle) -> Self {
//         ffi::CudaTriangle {
//             incenter: item.incenter.into(),
//             norm: item.norm.into(),
//             sides: [item.sides[0].into(),
//                     item.sides[1].into(),
//                     item.sides[2].into()],
//             side_lens: [item.side_lens[0] as f32,
//                         item.side_lens[1] as f32,
//                         item.side_lens[2] as f32]
//         }
//     }
// }

// impl RayCaster for CudaRayCaster {
//     fn project_ray(&self, r: &Ray, s: &Scene, ignore_objid: usize,
//                    depth: usize, runtimes: &mut HashMap<String, ProgressStat>) -> Color {

//         if depth == 0 {
//             return Vec3(0., 0., 0.);
//         }

//         let blue = Vec3(0.5, 0.7, 1.);
//         blue

//         // let t1 = raytrace::get_thread_time();

//         // let mut path: Vec<BoundingBox> = Vec::new();
//         // let hm_objs = s.boxes.get_all_objects_for_ray(&s.tris, r, &mut path);
//         // let cuda_objs: Vec<usize> = Vec::from_iter(hm_objs.iter().map(|i| i as usize));

//         // let t2 = raytrace::get_thread_time();
        
//         // let mut cuda_runtimes = [0 as u64; 3];
//         // let c = ffi::project_ray_cuda(&(*r).into(), &cuda_objs, ignore_objid, depth, &mut cuda_runtimes).into();
//         // let t3 = raytrace::get_thread_time();

//         // *runtimes.entry("BoundingBox".to_string()).or_default() += time::Duration::from_nanos((t2-t1) as u64);
//         // *runtimes.entry("Cuda Intersections".to_string()).or_default() += time::Duration::from_nanos((t3-t2) as u64);
//         // *runtimes.entry("Cuda Pre Memcpy".to_string()).or_default() += time::Duration::from_nanos(cuda_runtimes[0]);
//         // *runtimes.entry("Cuda Execute".to_string()).or_default() += time::Duration::from_nanos(cuda_runtimes[1]);
//         // *runtimes.entry("Cuda Post Memcpy".to_string()).or_default() += time::Duration::from_nanos(cuda_runtimes[2]);

//         // c
//     }

//     fn color_ray(&self, _r: &Ray, _s: &Scene, _objidx: usize,
//                 _point: &Point, _face: &CollisionFace, _depth: usize,
//                 _runtimes: &mut HashMap<String, ProgressStat>) -> Color {
//         raytrace::make_color((255, 0, 0))
//     }
// }

// fn preload_triangles(tris: &Vec<Triangle>) {
//     let cuda_tris: Vec<CudaTriangle> = Vec::from_iter(tris.iter().map(
//                         |t| {
//                             (*t).into()
//                         }));
//     ffi::preload_triangles_cuda(&cuda_tris);
// }

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
    v.walk_rays(&s, &mut data, 16, &caster, false);
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
    obj_data.extend(obj_parser::parse_obj("teapot_tri.obj", 10,
                                         &make_vec(&[0., 0.5, 5.]),
                                         1.0,
                                         raytrace::create_transform(&make_vec(&[0., 0.3, 1.]).unit(),
                                                                    270_f32.to_radians()),
                                        //  &SurfaceKind::Solid { color: make_color((252, 119, 0))},
                                         &SurfaceKind::Matte { color: make_color((252, 119, 0)),
                                                               alpha: 0.2 },
                                         0.05));

    // obj_data.extend(obj_parser::parse_obj("teapot_tri.obj", 10,
    //                                      &make_vec(&[1., -1.5, 4.]),
    //                                      1.0,
    //                                      raytrace::create_transform(&make_vec(&[0., 0.3, 1.]).unit(),
    //                                                                 270_f32.to_radians()),
    //                                      &SurfaceKind::Solid { color: make_color((200, 119, 80))},
    //                                      0.02));

    // let obj_data = make_sphere(&Vec3(2., 0., 5.), 1., (10, 30),
    //                            &SurfaceKind::Solid { color: make_color((0, 200, 100))},
    //                            0.);

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
                            //   &SurfaceKind::Solid { color: make_color((0, 252, 119))},
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
                            //   &SurfaceKind::Solid { color: make_color((0, 119, 252))},
                            //   &SurfaceKind::Solid { color: make_color((0, 119, 252))},
                              &SurfaceKind::Reflective { scattering: 0.002,
                                           color: make_color((230, 230, 230)),
                                           alpha: 0.7 },
                              &SurfaceKind::Matte { color: make_color((40, 40, 40)),
                                           alpha: 0.2 },
                              -1.));

    // optimize(&obj_data, &v, (10, 15));
    // return Ok(());

    // let bbox = raytrace::build_empty_box();
    // let bbox = raytrace::build_trivial_bounding_box(&obj_data,
    //                                         &make_vec(&[0., 0., 0.]),
    //                                         20.);
    let bbox = raytrace::build_bounding_box(&obj_data,
                                            &make_vec(&[0., 0., 20.1]),
                                            20.,
                                            7,
                                            19);


    // let light_orig = Vec3(7., 15., 4.);

    // preload_triangles(&obj_data);
    // bbox.print_tree();

    let s = Scene {
        tris: obj_data,
        boxes: bbox,
        // otherobjs: otherobjs,
        // lights: light
        // lights:  None
    };


    // let caster = CudaRayCaster {};
    let caster = DefaultRayCaster {};

    // let mut data = vec![make_vec(&[0., 0., 0.]); 1 as usize ];
    // let progress_ctx = v.walk_one_ray(&s, &mut data, (416, 130), &caster);
    // let _ = raytrace::write_png(file, (1, 1), &data);

    let mut data = vec![make_vec(&[0., 0., 0.]); (width*height) as usize ];
    let progress_ctx = v.walk_rays(&s, &mut data, 1, &caster, true);
    progress_ctx.print_stats();
    let _ = raytrace::write_png(file, (width, height), &data);

    return Ok(());
}
