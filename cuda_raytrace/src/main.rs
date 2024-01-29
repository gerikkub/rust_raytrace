
use raytrace_lib::raytrace::{make_color, make_polygon, BoundingBox, Collidable, CollisionFace, CollisionObject, Color, DefaultRayCaster, Disk, LightSource, Point, Ray, Scene, Sphere, SurfaceKind, Triangle, Vec3};
use raytrace_lib::obj_parser;
use raytrace_lib::raytrace;
use raytrace_lib::raytrace::RayCaster;

use core::time;
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
        side_lens: [f32; 3],
        id: usize
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


impl From<Vec3> for ffi::CudaVec3 {
    fn from(item: Vec3) -> Self {
        ffi::CudaVec3 {v: [item.0 as f32, item.1 as f32, item.2 as f32]}
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

impl Into<Color> for ffi::CudaColor {
    fn into(self) -> Color {
        make_color((self.c[0], self.c[1], self.c[2]))
    }
}

impl From<Triangle> for ffi::CudaTriangle {
    fn from(item: Triangle) -> Self {
        ffi::CudaTriangle {
            incenter: item.incenter.into(),
            norm: item.norm.into(),
            sides: [item.sides[0].into(),
                    item.sides[1].into(),
                    item.sides[2].into()],
            side_lens: [item.side_lens[0] as f32,
                        item.side_lens[1] as f32,
                        item.side_lens[2] as f32],
            id: item.id
        }
    }
}

impl RayCaster for CudaRayCaster {
    fn project_ray(&self, r: &Ray, s: &Scene, ignore_objid: usize,
                   depth: usize, runtimes: &mut HashMap<String, time::Duration>) -> Color {

        if depth == 0 {
            return Vec3(0., 0., 0.);
        }

        let blue = Vec3(0.5, 0.7, 1.);

        let t1 = raytrace::get_thread_time();

        let mut path: Vec<BoundingBox> = Vec::new();
        let hm_objs = s.boxes.get_all_objects_for_ray(&s.tris, r, &mut path);
        let cuda_objs: Vec<usize> = Vec::from_iter(hm_objs.iter().map(|i| i as usize));

        let t2 = raytrace::get_thread_time();
        
        let mut cuda_runtimes = [0 as u64; 3];
        let c = ffi::project_ray_cuda(&(*r).into(), &cuda_objs, ignore_objid, depth, &mut cuda_runtimes).into();
        let t3 = raytrace::get_thread_time();

        *runtimes.entry("BoundingBox".to_string()).or_default() += time::Duration::from_nanos((t2-t1) as u64);
        *runtimes.entry("Cuda Intersections".to_string()).or_default() += time::Duration::from_nanos((t3-t2) as u64);
        *runtimes.entry("Cuda Pre Memcpy".to_string()).or_default() += time::Duration::from_nanos(cuda_runtimes[0]);
        *runtimes.entry("Cuda Execute".to_string()).or_default() += time::Duration::from_nanos(cuda_runtimes[1]);
        *runtimes.entry("Cuda Post Memcpy".to_string()).or_default() += time::Duration::from_nanos(cuda_runtimes[2]);

        c
    }

    fn color_ray(&self, _r: &Ray, _s: &Scene, _objidx: usize,
                _point: &Point, _face: &CollisionFace, _depth: usize,
                _runtimes: &mut HashMap<String, time::Duration>) -> Color {
        raytrace::make_color((255, 0, 0))
    }
}

fn preload_triangles(tris: &Vec<Triangle>) {
    let cuda_tris: Vec<CudaTriangle> = Vec::from_iter(tris.iter().map(
                        |t| {
                            (*t).into()
                        }));
    ffi::preload_triangles_cuda(&cuda_tris);
}


fn main() -> Result<()> {

    let path = Path::new("test.png");
    let file = fs::File::create(&path)?;

    let aspect = 2160. / 3840.;
    let width = 3840;
    let height = 2160;

    // let aspect = 1440. / 2560.;
    // let width = 2560;
    // let height = 1440;

    // let aspect = 480. / 640.;
    // let width = 640;
    // let height = 480;

    // let aspect = 480. / 640.;
    // let width = 160;
    // let height = 120;

    let mut data = vec![Vec3(0., 0., 0.); (width*height) as usize ];

    let obj_data = obj_parser::parse_obj("teapot_tri.obj", 10,
                                         &Vec3(0., 0.5, 5.),
                                         raytrace::create_transform(&Vec3(0., 0.3, 1.).unit(),
                                                                    270_f64.to_radians()),
                                         &SurfaceKind::Solid { color: make_color((252, 119, 0))} );
                                        //  &SurfaceKind::Matte { color: make_color((252, 119, 0)),
                                        //                        alpha: 0.2 });

    // let mut obj_data: Vec<Triangle> = Vec::new();

    // obj_data.push(make_polygon(&[Vec3(1., 1.5, 3.5),
    //                              Vec3(1., 3., 4.),
    //                              Vec3(3., 1.5, 4.)],
    //                              &SurfaceKind::Solid { color: make_color((22, 220, 20)) },
    //                              0.99, 4));


    let v = raytrace::create_viewport((width, height),
                            (1., 1. * aspect),
                            &Vec3(2., 0., 0.),
                            &Vec3(0., 0., 1.).unit(),
                            90.,
                            0_f64.to_radians(),
                            2,
                            100);
    
    println!("Viewport: {:?}", v);
    
    // let bbox = raytrace::build_empty_box();
    // let bbox = raytrace::build_trivial_bounding_box(&obj_data.objs,
    //                                         &Vec3(0., 0., 0.),
    //                                         20.);
    let bbox = raytrace::build_bounding_box(&obj_data.objs,
                                            &Vec3(0., 0., 0.),
                                            20.,
                                            40,
                                            10);

    let mut otherobjs: Vec<CollisionObject> = Vec::new();


    otherobjs.push(CollisionObject::Disk(Disk {
        orig: Vec3(4., 4., 7.),
        norm: Vec3(-0.3, -0.55, -0.5).unit(),
        r: 2.,
        depth: 0.1,
        surface: SurfaceKind::Reflective { scattering: 0.002,
                                           color: make_color((230, 230, 230)),
                                           alpha: 0.7 },
        side_surface: SurfaceKind::Matte { color: make_color((40, 40, 40)),
                                           alpha: 0.2 },
        id: 1
    }));

    otherobjs.push(CollisionObject::Sphere(Sphere {
        orig: Vec3(-50., 0., 5.),
        r: 50.,
        surface: SurfaceKind::Matte { color: make_color((40, 40, 40)),
                                      alpha: 0.2 },
        id: 2
    }));

    otherobjs.push(CollisionObject::Disk(Disk {
        orig: Vec3(4., -3., 5.),
        norm: Vec3(-0.5, 1.0, -0.5).unit(),
        r: 1.,
        depth: 0.04,
        surface: SurfaceKind::Reflective { scattering: 0.002,
                                           color: make_color((230, 230, 230)),
                                           alpha: 0.7 },
        side_surface: SurfaceKind::Matte { color: make_color((40, 40, 40)),
                                           alpha: 0.2 },
        id: 3
    }));

    // let poly = make_polygon(&vec![Vec3(1., -1., 3.),
    //                         Vec3(1., 1., 3.),
    //                         Vec3(3., 1., 3.),
    //                         Vec3(3., -1., 3.)],
    //                         &SurfaceKind::Solid { color: make_color((220, 20, 20)) },
    //                         0.99, 3);

    // otherobjs.push(CollisionObject::ConvexPolygon(poly));

    // otherobjs.push(CollisionObject::Triangle(make_triangle((Vec3(1., 1.5, 3.5),
    //                                                         Vec3(1., 3., 4.),
    //                                                         Vec3(3., 1.5, 4.)),
    //                                                        &SurfaceKind::Solid { color: make_color((22, 220, 20)) },
    //                                                        0.99, 4)));

    let light_orig = Vec3(7., 15., 4.);

    // otherobjs.push(CollisionObject::Sphere(Sphere {
    //     orig: light_orig,
    //     r: 1.0,
    //     surface: SurfaceKind::Solid { color: make_color((255, 255, 255)) },
    //     id: 3
    // }));

    let light = LightSource {
        orig: light_orig,
        len2: 0.01
    };

    preload_triangles(&obj_data.objs);

    let s = Scene {
        tris: obj_data.objs,
        boxes: bbox,
        // otherobjs: otherobjs,
        // lights: light
        // lights:  None
    };


    // let caster = CudaRayCaster {};
    let caster = DefaultRayCaster {};

    let progress_ctx = v.walk_rays(&s, &mut data, 16, &caster);

    progress_ctx.print_stats();

    let _ = raytrace::write_png(file, (width, height), &data);
    return Ok(());
}
