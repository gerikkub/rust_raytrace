#![feature(iter_collect_into)]

use raytrace_lib::raytrace::{make_vec, make_color, make_disk, BoundingBox, BBSubobj, Collidable,
                             Triangle, Color, Ray, Scene, SurfaceKind, Vec3, Viewport};
use raytrace_lib::obj_parser;
use raytrace_lib::raytrace;
use raytrace_lib::raytrace::RayCaster;
use raytrace_lib::progress::ProgressStat;
use raytrace_lib::debug;

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc::Sender;
use std::collections:: BTreeMap;
use ordered_float::OrderedFloat;
use std::iter::Iterator;
use std::time;

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
        include!("cuda_raytrace_lib/src/cuda_rt.h");

        fn exec_cuda_raytrace<'a>(alltris: &Vec<CudaTriangle>,
                                  rays: &Vec<CudaRay>,
                                  tris: &Vec<u32>,
                                  trilist_stride: u32,
                                  stream_num: u32,
                                  hit_nums: &mut [u32],
                                  hit_times: &mut [f32],
                                  runtimes: &mut [u64; 4]);
    }

}

pub struct CudaRayCaster {
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
                      hit_times: &mut [f32],
                      _runtimes: &mut [u64; 4]) {

    // print!("RUST Exec\n");
    // print!(" Rays: {}\n", cudarays.len());
    // print!(" Triangles: {}\n", tris.len());
    // print!(" Triangle Stride: {}\n", trilist_stride);

    let mut alltris: Vec<Triangle> = Vec::new();

    for (num, ct) in cudatris.iter().enumerate() {
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
            edge_thickness: 0.1,
            num: num
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
            hit_times[r_idx] = 0.;
        } else {
            let min_hit: &(f32, u32) = hits.iter().reduce(|best, h| if h.0 < best.0 { h } else { best}).unwrap();
            hit_nums[r_idx] = min_hit.1;
            hit_times[r_idx] = min_hit.0;
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
    next_key: f32,
    best_hit: Option<(f32, usize)>
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
                        cols: &mut dyn Iterator<Item = usize>,
                        row: usize,
                        stream_num: u32,
                        data: &mut [Color],
                        runstats: &mut HashMap<String, ProgressStat>) {

    // println!("Row {}", row);

    runstats.insert("Cycle Prep".to_string(), ProgressStat::Time(time::Duration::from_nanos(0)));
    runstats.insert("Cycle Post".to_string(), ProgressStat::Time(time::Duration::from_nanos(0)));

    runstats.insert("CUDA Mem Init".to_string(), ProgressStat::Time(time::Duration::from_nanos(0)));
    runstats.insert("CUDA Host->Device Memcpy".to_string(), ProgressStat::Time(time::Duration::from_nanos(0)));
    runstats.insert("CUDA Execute".to_string(), ProgressStat::Time(time::Duration::from_nanos(0)));
    runstats.insert("CUDA Device->Host Execute".to_string(), ProgressStat::Time(time::Duration::from_nanos(0)));
    runstats.insert("CUDA Threads".to_string(), ProgressStat::Count(0));

    runstats.insert("Rays".to_string(), ProgressStat::Count(0));


    let blue = make_color((128, 178, 255));
    let red = make_color((255, 0, 0));

    let t0 = time::Instant::now();

    let mut workqueue: VecDeque<WorkQueueEntry> = VecDeque::new();
    let mut results: Vec<Vec<Color>> = Vec::new();
    for _ in 0..v.width {
        results.push(Vec::new());
    }

    for col in &mut *cols {
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

            if s.debug_en {
                let mut debug_ctx = s.debug_ctx.lock().unwrap();
                debug_ctx.register_ray(&r, (row, col));
                debug_ctx.add_ray(&r);
            }
        }
    }

    let t1 = time::Instant::now();
    runstats.insert("Work Prep".to_string(), ProgressStat::Time(t1-t0));

    let mut workcycle: VecDeque<WorkCycleEntry> = VecDeque::new();
    let mut nextcycle: VecDeque<WorkCycleEntry> = VecDeque::new();

    let mut prep_time: time::Duration = time::Duration::from_nanos(0);

    while !(workqueue.len() == 0 && nextcycle.len() == 0) {

        let prep_t0 = time::Instant::now();

        workcycle.extend(nextcycle.drain(0..));

        while workcycle.len() < 512 {
            if workqueue.len() == 0 {
                break;
            }
            let workitem = workqueue.pop_front().unwrap();

            let (next_f32, item_tris) = get_tris_for_btree(&workitem.tri_map, 0., tri_stride);

            let e = WorkCycleEntry {
                col: workitem.col,
                sample: workitem.sample,
                ray: workitem.ray,
                tri_ctx: TriMapCtx {
                    tri_map: workitem.tri_map,
                    cycle_tris: item_tris,
                    next_key: next_f32,
                    best_hit: None
                }
            };
            workcycle.push_back(e);
        }

        if s.debug_en {
            let mut debug_ctx = s.debug_ctx.lock().unwrap();
            for item in &workcycle {
                debug_ctx.update_ray_triangles(&item.ray, &item.tri_ctx.cycle_tris);
            }
        }

        let workrays: Vec<CudaRay> = workcycle.iter().map(|e| (e.ray).into()).collect();
        let mut worktris = vec![0 as u32; workrays.len() * tri_stride];
        for (r_idx, e) in workcycle.iter().enumerate() {
            let tris_temp: Vec<u32> = e.tri_ctx.cycle_tris.iter().map(|t| *t as u32).collect();
            worktris.as_mut_slice()[r_idx * tri_stride..((r_idx) * tri_stride) + tris_temp.len()]
                    .copy_from_slice(tris_temp.as_slice());
        }

        let mut hit_nums = vec![0 as u32; workrays.len()];
        let mut hit_times = vec![0. as f32; workrays.len()];
        let mut runtimes = [0 as u64; 4];

        exec_rust_raytrace(cudatris,
                           &workrays,
                        //    &workcycle.iter().map(|e| (e.ray).into()).collect(),
                           &worktris,
                           tri_stride,
                           stream_num,
                           &mut hit_nums,
                           &mut hit_times,
                           &mut runtimes);

        let prep_t1 = time::Instant::now();

        let mut cuda_hit_nums = vec![0 as u32; workrays.len()];
        let mut cuda_hit_times = vec![0. as f32; workrays.len()];
        let mut cuda_runtimes = [0 as u64; 4];

        exec_cuda_raytrace(cudatris,
                           &workrays,
                           &worktris,
                           tri_stride as u32,
                           stream_num,
                           &mut cuda_hit_nums,
                           &mut cuda_hit_times,
                           &mut cuda_runtimes);

        for idx in 0..workrays.len() {
            if hit_nums[idx] != cuda_hit_nums[idx] {
                println!("RUST/CUDA Hit diff at ({},{}). {} {} vs {} {}",
                         row,
                         workcycle[idx].col,
                         hit_nums[idx], hit_times[idx],
                         cuda_hit_nums[idx], cuda_hit_times[idx]);
            }
        }

        let prep_t2 = time::Instant::now();

        for (hit, t) in cuda_hit_nums.iter().zip(cuda_hit_times) {
            let workitem = workcycle.pop_front().unwrap();

            let best = 
                if *hit != 0 {
                    Some(match workitem.tri_ctx.best_hit {
                        Some (tb) => {
                            if t < tb.0 { (t, *hit as usize) } else { tb }
                        },
                        None => (t, *hit as usize)
                    })
                } else {
                    workitem.tri_ctx.best_hit
                };

            if workitem.tri_ctx.next_key < f32::MAX {

                if best.is_none() || best.unwrap().0 > workitem.tri_ctx.next_key || true {
                    // Need to keep iterating

                    let (next_f32, item_tris) = get_tris_for_btree(&workitem.tri_ctx.tri_map, workitem.tri_ctx.next_key, tri_stride);
                    let e = WorkCycleEntry {
                        col: workitem.col,
                        sample: workitem.sample,
                        ray: workitem.ray,
                        tri_ctx: TriMapCtx {
                            tri_map: workitem.tri_ctx.tri_map,
                            cycle_tris: item_tris,
                            next_key: next_f32,
                            best_hit: best
                        }
                    };
                    nextcycle.push_front(e);
                } else {
                    results[workitem.col].push(match best {
                        Some(_t) => red,
                        None => blue
                    });

                    if s.debug_en {
                        let mut debug_ctx = s.debug_ctx.lock().unwrap();
                        match best {
                            Some ((t, hit)) => {
                                debug_ctx.update_ray_hit(&workitem.ray, hit, t);
                            },
                            None => {}
                        }
                    }
                }
            } else {
                results[workitem.col].push(match best {
                    Some(_t) => red,
                    None => blue
                });
                if s.debug_en {
                    let mut debug_ctx = s.debug_ctx.lock().unwrap();
                    match best {
                        Some ((t, hit)) => {
                            debug_ctx.update_ray_hit(&workitem.ray, hit, t);
                        },
                        None => {}
                    }
                }
            }
        }

        let prep_t3 = time::Instant::now();

        *runstats.get_mut("Cycle Prep").unwrap() += ProgressStat::Time(prep_t1-prep_t0);
        *runstats.get_mut("Cycle Post").unwrap() += ProgressStat::Time(prep_t3-prep_t2);

        *runstats.get_mut("CUDA Mem Init").unwrap() += ProgressStat::Time(time::Duration::from_nanos(cuda_runtimes[0]));
        *runstats.get_mut("CUDA Host->Device Memcpy").unwrap() += ProgressStat::Time(time::Duration::from_nanos(cuda_runtimes[1]));
        *runstats.get_mut("CUDA Execute").unwrap() += ProgressStat::Time(time::Duration::from_nanos(cuda_runtimes[2]));
        *runstats.get_mut("CUDA Device->Host Execute").unwrap() += ProgressStat::Time(time::Duration::from_nanos(cuda_runtimes[3]));
        *runstats.get_mut("CUDA Threads").unwrap() += ProgressStat::Count(workrays.len() * tri_stride);

        *runstats.get_mut("Rays").unwrap() += ProgressStat::Count(workrays.len());

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

impl CudaRayCaster {

    pub fn cast_one_ray(&self, v: &Viewport, s: &Scene,
                        pixel: (usize, usize),
                        tri: usize) {

        let mut rays: Vec<Ray> = Vec::new();
        rays.push(v.pixel_ray(pixel));

        let mut tris: Vec<u32> = Vec::new();
        tris.push(tri as u32);
        let mut hit_num = [0 as u32; 1];
        let mut hit_time = [0. as f32; 1];
        let mut runtimes = [0 as u64; 4];
        // exec_rust_raytrace(&s.tris,
        //                    &rays,
        //                    &tris,
        //                    1,
        //                    0,
        //                    &mut hit_num,
        //                    &mut hit_time,
        //                    &mut runtimes);

        println!("Hit {} at {}", hit_num[0], hit_time[0]);
    }
}

impl RayCaster for CudaRayCaster {

    fn walk_rays_internal(&self, v: &Viewport, s: &Scene,
                          data: & mut[Color], _threads: usize,
                          progress_tx: Sender<(usize, usize, usize, HashMap<String, ProgressStat>)>) {

        println!("Starting CUDA Raycaster");
        let blue = make_color((128, 178, 255));

        for idx in 0..data.len() {
            data[idx] = blue;
        }

        for row in 0..v.height {

            let mut runstats: HashMap<String, ProgressStat> = HashMap::new();

            walk_rays_workqueue(v, s, &s.tris,
                                &s.tris.iter().map(|t| (*t).into()).collect(),
                                256,
                                &mut (0..v.width),
                                row,
                                0,
                                &mut data[row * v.width..(row+1) * v.width],
                                &mut runstats);
            // break;
            let _ = progress_tx.send((0, row, v.width, runstats));
        }
    }
}