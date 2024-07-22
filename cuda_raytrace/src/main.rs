#![feature(iter_collect_into)]

use raytrace_lib::raytrace::{make_vec, make_color, make_disk, BoundingBox, BBSubobj, Collidable,
                             Triangle, Color, Ray, Scene, SurfaceKind, Vec3, Viewport};
use raytrace_lib::obj_parser;
use raytrace_lib::raytrace;
use raytrace_lib::raytrace::RayCaster;
use raytrace_lib::progress::ProgressStat;

use std::env;
use std::time;
use std::fs;
use std::io::Result;
use std::path::Path;
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::Sender;
use std::collections:: BTreeMap;
use ordered_float::OrderedFloat;

use sdl2;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;

use ffi::{exec_cuda_raytrace, CudaRay, CudaTriangle};

#[cxx::bridge]
mod ffi {

    struct CudaVec3 {
        v: [f32; 3]
    }

    struct CudaRay {
        a: CudaVec3,
        u: CudaVec3
    }

    struct CudaTriangle {
        incenter: CudaVec3,
        norm: CudaVec3,
        sides: [CudaVec3; 3]
    }


    extern "Rust" {

        type CudaRayCaster;

    }

    unsafe extern "C++" {
        include!("cuda_raytrace/src/cuda_rt.h");

        fn exec_cuda_raytrace<'a>(alltris: &Vec<CudaTriangle>,
                                  rays: &Vec<CudaRay>,
                                  tris: &Vec<u32>,
                                  trilist_stride: u32,
                                  stream_num: u32,
                                  hit_nums: &mut [u32],
                                  runtimes: &mut [u64; 4]);
    }

}

struct CudaRayCaster {
}


impl From<Vec3> for ffi::CudaVec3 {
    fn from(item: Vec3) -> Self {
        ffi::CudaVec3 {v: [item.v[0], item.v[1], item.v[2]]}
    }
}

impl From<Ray> for ffi::CudaRay {
    fn from(item: Ray) -> Self {
        ffi::CudaRay {
            a: item.orig.into(),
            u: item.dir.into()
        }
    }
}

impl From<Triangle> for ffi::CudaTriangle {
    fn from(item: Triangle) -> Self {
        ffi::CudaTriangle {
            incenter: item.incenter.into(),
            norm: item.norm.into(),
            sides: [item.sides[0].mult(item.side_lens[0]).into(),
                    item.sides[1].mult(item.side_lens[1]).into(),
                    item.sides[2].mult(item.side_lens[2]).into()]
        }
    }
}

// impl From<BoundingBox> for ffi::CudaBoundingBox {
//     fn from(item: BoundingBox) -> Self {
//         match &item.objs {
//             BBSubobj::Boxes(subboxes) => {
//                 ffi::CudaBoundingBox {
//                     orig: item.orig.into(),
//                     len2: item.len2,
//                     kind: CudaBBSubobj::Boxes,
//                     subboxes: subboxes.iter().map(|b| b.clone().into()).collect(),
//                     subobjs: Vec::new(),
//                     depth: item.depth
//                 }
//             },
//             BBSubobj::Tris(subtris) => {
//                 ffi::CudaBoundingBox {
//                     orig: item.orig.into(),
//                     len2: item.len2,
//                     kind: CudaBBSubobj::Tris,
//                     subboxes: Vec::new(),
//                     subobjs: subtris.clone(),
//                     depth: item.depth
//                 }
//             }
//         }
//     }
// }

fn exec_rust_raytrace(cudatris: &Vec<CudaTriangle>,
                      cudarays: &Vec<CudaRay>,
                      tris: &Vec<u32>,
                      trilist_stride: usize,
                      _stream_num: u32,
                      hit_nums: &mut [u32],
                      _runtimes: &mut [u64; 4]) {

    print!("RUST Exec\n");
    print!(" Rays: {}\n", cudarays.len());
    print!(" Triangles: {}\n", tris.len());
    print!(" Triangle Stride: {}\n", trilist_stride);

    let mut alltris: Vec<Triangle> = Vec::new();

    for ct in cudatris {
        // let s1_len2 = make_vec(&[ct.sides[0].v[0],
        //                          ct.sides[0].v[1],
        //                          ct.sides[0].v[2]]).len2();
        // let s2_len2 = make_vec(&[ct.sides[1].v[0],
        //                          ct.sides[1].v[1],
        //                          ct.sides[1].v[2]]).len2();
        // let s3_len2 = make_vec(&[ct.sides[2].v[0],
        //                          ct.sides[2].v[1],
        //                          ct.sides[2].v[2]]).len2();
        // let b_r2 = s1_len2.max(s2_len2).max(s3_len2);
        let b_r2 = 1000.;
        let t = Triangle {
            incenter: make_vec(&[ct.incenter.v[0],
                                 ct.incenter.v[1],
                                 ct.incenter.v[2]]),
            norm: make_vec(&[ct.norm.v[0],
                             ct.norm.v[1],
                             ct.norm.v[2]]),
            bounding_r2: b_r2,
            sides: [
                make_vec(&[ct.sides[0].v[0],
                           ct.sides[0].v[1],
                           ct.sides[0].v[2]]).unit(),
                make_vec(&[ct.sides[1].v[0],
                           ct.sides[1].v[1],
                           ct.sides[1].v[2]]).unit(),
                make_vec(&[ct.sides[2].v[0],
                           ct.sides[2].v[1],
                           ct.sides[2].v[2]]).unit(),
            ],
            side_lens: [
                make_vec(&[ct.sides[0].v[0],
                           ct.sides[0].v[1],
                           ct.sides[0].v[2]]).len(),
                make_vec(&[ct.sides[1].v[0],
                           ct.sides[1].v[1],
                           ct.sides[1].v[2]]).len(),
                make_vec(&[ct.sides[2].v[0],
                           ct.sides[2].v[1],
                           ct.sides[2].v[2]]).len(),
            ],
            corners: [
                make_vec(&[0., 0., 0.]),
                make_vec(&[0., 0., 0.]),
                make_vec(&[0., 0., 0.])
            ],
            surface: SurfaceKind::Solid { color: make_color((255, 0, 0)) },
            edge_thickness: 0.1
        };
        alltris.push(t);
    }

    let mut rays: Vec<Ray> = Vec::new();

    for cr in cudarays {
        let r = Ray {
            orig: make_vec(&[cr.a.v[0],
                             cr.a.v[1],
                             cr.a.v[2]]),
            dir: make_vec(&[cr.u.v[0],
                            cr.u.v[1],
                            cr.u.v[2]]),
            inv_dir: make_vec(&[1./cr.u.v[0],
                                1./cr.u.v[1],
                                1./cr.u.v[2]]),
        };
        rays.push(r);
    }

    for (r_idx, r) in rays.iter().enumerate() {

        let mut hits: Vec<(f32, u32)> = Vec::new();
        for tri_num in tris.as_slice()[r_idx*trilist_stride..(r_idx+1)*trilist_stride].iter() {
            let t = alltris[*tri_num as usize];

            let hit = t.intersects(r);
            match hit {
                Some((time, _p, _face)) => {
                    hits.push((time, *tri_num));
                },
                None => {}
            };
        }

        if hits.len() == 0 {
            hit_nums[r_idx] = 0;
        } else {
            let min_hit: &(f32, u32) = hits.iter().reduce(|best, h| if h.0 < best.0 { h } else { best}).unwrap();
            hit_nums[r_idx] = min_hit.1;
        }
    }
}

fn get_tris_for_btree(obj: &BTreeMap<OrderedFloat<f32>, Vec<usize>>,
                      min_key: f32,
                      max_tris: usize) -> (f32, Vec<usize>) {
    let mut tris: Vec<usize> = Vec::new();
    let mut next_key = min_key;

    for k in obj.keys().filter(|k| **k > OrderedFloat(min_key)) {
        assert!(obj[k].len() < max_tris);
        next_key = (*k).into();

        if tris.len() + obj[k].len() > max_tris {
            break;
        }

        tris.extend(obj[k].clone());
    };

    if obj.keys().filter(|k| **k > OrderedFloat(next_key)).count() == 0 {
        next_key = f32::MAX;
    }

    (next_key, tris)
}

struct WorkQueueEntry {
    col: usize,
    sample: usize,
    ray: Ray,
    tri_map: BTreeMap<OrderedFloat<f32>, Vec<usize>>
}

struct TriMapCtx {
    tri_map: BTreeMap<OrderedFloat<f32>, Vec<usize>>,
    cycle_tris: Vec<usize>,
    next_key: f32
}

struct WorkCycleEntry {
    col: usize,
    sample: usize,
    ray: Ray,
    tri_ctx: TriMapCtx
}


fn walk_rays_workqueue(v: &Viewport, s: &Scene,
                        tris: &Vec<Triangle>,
                        cudatris: &Vec<CudaTriangle>,
                        tri_stride: usize,
                        row: usize,
                        stream_num: u32,
                        data: &mut [Color]) {

    println!("Row {}", row);

    let blue = make_color((128, 178, 255));
    let red = make_color((255, 0, 0));

    let mut workqueue: VecDeque<WorkQueueEntry> = VecDeque::new();
    let mut results: Vec<Vec<Color>> = Vec::new();
    for _ in 0..v.width {
        results.push(Vec::new());
    }

    for col in 0..v.width {
        for sample in 0..v.samples_per_pixel {
            let r = v.pixel_ray((row, col));
            let objs = s.boxes.get_all_objects_for_ray(tris, &r);
            let e = WorkQueueEntry {
                col: col,
                sample: sample,
                ray: r,
                tri_map: objs
            };
            workqueue.push_back(e);
        }
    }

    let mut workcycle: VecDeque<WorkCycleEntry> = VecDeque::new();
    let mut nextcycle: VecDeque<WorkCycleEntry> = VecDeque::new();

    while !(workqueue.len() == 0 && nextcycle.len() == 0) {

        workcycle.extend(nextcycle.drain(0..));

        while workcycle.len() < 64 {
            if workqueue.len() == 0 {
                break;
            }
            let workitem = workqueue.pop_front().unwrap();

            let (next_f32, item_tris) = get_tris_for_btree(&workitem.tri_map, 0., tri_stride);

            // println!("Ray {:#?}", workitem.ray);

            let e = WorkCycleEntry {
                col: workitem.col,
                sample: workitem.sample,
                ray: workitem.ray,
                tri_ctx: TriMapCtx {
                    tri_map: workitem.tri_map,
                    cycle_tris: item_tris,
                    next_key: next_f32
                }
            };
            workcycle.push_back(e);
        }

        let workrays: Vec<CudaRay> = workcycle.iter().map(|e| (e.ray).into()).collect();
        let mut worktris = vec![0 as u32; workrays.len() * tri_stride];
        for (r_idx, e) in workcycle.iter().enumerate() {
            let tris_temp: Vec<u32> = e.tri_ctx.cycle_tris.iter().map(|t| *t as u32).collect();
            worktris.as_mut_slice()[r_idx * tri_stride..((r_idx) * tri_stride) + tris_temp.len()]
                    .copy_from_slice(tris_temp.as_slice());
            // println!("R {} Tris: {} {} {} {}", r_idx,
            //          worktris[r_idx*tri_stride],
            //          worktris[r_idx*tri_stride + 1],
            //          worktris[r_idx*tri_stride + 2],
            //          worktris[r_idx*tri_stride + 3]);
                     
        }

        let mut hit_nums = vec![0 as u32; workrays.len()];
        let mut cuda_runtimes = [0 as u64; 4];

        // exec_rust_raytrace(cudatris,
        //                    &workrays,
        //                    &worktris,
        //                    tri_stride,
        //                    stream_num,
        //                    &mut hit_nums,
        //                    &mut cuda_runtimes);

        exec_cuda_raytrace(cudatris,
                           &workrays,
                           &worktris,
                           tri_stride as u32,
                           stream_num,
                           &mut hit_nums,
                           &mut cuda_runtimes);


        for hit in hit_nums {
            let workitem = workcycle.pop_front().unwrap();
            if hit != 0 {
                // let c = match tris[(*hit) as usize].surface {
                //     SurfaceKind::Solid {color} => color,
                //     SurfaceKind::Matte {color, alpha} => color,
                //     SurfaceKind::Reflective {scattering, color, alpha} => color
                // };
                // results[workitem.0].push(c);
                // println!("Push Results hit {} {}", workitem.col, hit);
                results[workitem.col].push(red);
            } else {
                if workitem.tri_ctx.next_key < f32::MAX {

                    let (next_f32, item_tris) = get_tris_for_btree(&workitem.tri_ctx.tri_map, workitem.tri_ctx.next_key, tri_stride);
                    let e = WorkCycleEntry {
                        col: workitem.col,
                        sample: workitem.sample,
                        ray: workitem.ray,
                        tri_ctx: TriMapCtx {
                            tri_map: workitem.tri_ctx.tri_map,
                            cycle_tris: item_tris,
                            next_key: next_f32
                        }
                    };
                    nextcycle.push_front(e);
                } else {
                    // println!("Push Results miss {}", workitem.col);
                    results[workitem.col].push(blue);
                }
            }
        }

        // return;
    }

    for (c_idx, c_vec) in results.iter().enumerate() {
        assert!(c_vec.len() == 1);
        // let mut c_final = make_vec(&[0., 0., 0.]);
        // for c in c_vec {
        //     c_final.add(c);
        // };
        // data[c_idx] = c_final.mult(1./(v.samples_per_pixel as f32));
        data[c_idx] = c_vec[0];
    }
}


impl RayCaster for CudaRayCaster {




    fn walk_rays_internal(&self, v: &Viewport, s: &Scene,
                          data: & mut[Color], _threads: usize,
                          _progress_tx: Sender<(usize, usize, usize, HashMap<String, ProgressStat>)>) {

        println!("Starting CUDA Raycaster");
        let blue = make_color((128, 178, 255));

        for idx in 0..data.len() {
            data[idx] = blue;
        }

        // let t1 = raytrace::get_thread_time();

        for row in 0..v.height {
            walk_rays_workqueue(v, s, &s.tris,
                                &s.tris.iter().map(|t| (*t).into()).collect(),
                                1024,
                                row,
                                0,
                                &mut data[row * v.width..(row+1) * v.width]);
            // break;
        }

        // let t2 = raytrace::get_thread_time();

        // let mut runstats: hashmap<string, progressstat> = hashmap::new();

        // runstats.insert("cuda total".to_string(), progressstat::time(time::duration::from_nanos((t2-t1) as u64)));
        // runstats.insert("cuda pre memcpy".to_string(), progressstat::time(time::duration::from_nanos(cuda_runtimes[0])));
        // runstats.insert("cuda execute".to_string(), progressstat::time(time::duration::from_nanos(cuda_runtimes[1])));
        // runstats.insert("cuda post memcpy".to_string(), progressstat::time(time::duration::from_nanos(cuda_runtimes[2])));
        // runstats.insert("cuda color mapping".to_string(), progressstat::time(time::duration::from_nanos(cuda_runtimes[3])));
    
        // runstats.insert("rays".to_string(), progressstat::count(v.width*v.height));

        // let _ = progress_tx.send((0, 0, v.width*v.height, runstats));
    }
}

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

    let caster = CudaRayCaster {};

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

    // let aspect = 1. / 1.;
    // let width = 64;
    // let height = 64;

    // let aspect = 1. / 1.;
    // let width = 1;
    // let height = 1;

    let mut obj_data: Vec<Triangle> = Vec::new();
    obj_data.extend(obj_parser::parse_obj("teapot_tri.obj",
                                         &make_vec(&[0., 0.5, 5.]),
                                         1.0,
                                         raytrace::create_transform(&make_vec(&[0., 0.3, 1.]).unit(),
                                                                    270_f32.to_radians()),
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
                            1);

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

    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        let dbg_tri = args[1].parse::<usize>().unwrap();
        println!("Triangle {}\n{:#?}", dbg_tri, obj_data[dbg_tri]);
        return Ok(());
    }

    // obj_data.extend(make_disk(&make_vec(&[2., 0., 0.5]),
    //                           &make_vec(&[-1., 0., 0.]).unit(),
    //                           5.,
    //                           0.04,
    //                           50,
    //                           &SurfaceKind::Reflective { scattering: 0.002,
    //                                        color: make_color((230, 230, 230)),
    //                                        alpha: 0.7 },
    //                           &SurfaceKind::Matte { color: make_color((40, 40, 40)),
    //                                        alpha: 0.2 },
    //                           -1.));


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

    let s = Scene {
        tris: obj_data,
        boxes: bbox,
    };

    let caster = CudaRayCaster {};

    let mut data = vec![make_vec(&[0., 0., 0.]); (width*height) as usize ];
    let progress_ctx = caster.walk_rays(&v, &s, &mut data, 1, false);


    progress_ctx.print_stats();
    let _ = raytrace::write_png(file, (width, height), &data);

    if v.height == 1 {
        return Ok(());
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


    return Ok(());
}
