
use crate::raytrace::{Ray, Triangle};
use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;
use std::io;

#[derive(Clone, Debug)]
struct RayDebugCtx {
    ray: Ray,
    pixel: (usize, usize),
    check_tris: Vec<usize>,
    tri_hit: usize,
    hit_t: f32
}

impl fmt::Display for RayDebugCtx {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Pixel_x;Pixel_y;ray_p;ray_v;tri_hit;hit_t;check_tris;
        write!(f, "{};{};{},{},{};{},{},{};{};{};{}",
               self.pixel.0, self.pixel.1,
               self.ray.orig.v[0],
               self.ray.orig.v[1],
               self.ray.orig.v[2],
               self.ray.dir.v[0],
               self.ray.dir.v[1],
               self.ray.dir.v[2],
               self.tri_hit,
               self.hit_t,
               self.check_tris.iter().map(|t| t.to_string()).reduce(|acc, t| acc + "," + &t).unwrap()
        )
    }
}

pub struct DebugCtx {
    checks: HashMap<(usize,usize), RayDebugCtx>,
    rays: HashMap<(i64, i64, i64), (usize, usize)>
}

pub fn make_debug_ctx() -> DebugCtx {
    DebugCtx {
        checks: HashMap::new(),
        rays: HashMap::new()
    }
}

impl DebugCtx {

    pub fn register_ray(&mut self, ray: &Ray, pixel: (usize, usize)) {
        let r_x = (ray.orig.v[0] * (1000.)) as i64;
        let r_y = (ray.orig.v[1] * (1000.)) as i64;
        let r_z = (ray.orig.v[2] * (1000.)) as i64;

        self.rays.insert((r_x, r_y, r_z), pixel);
    }

    fn get_pixel_for_ray(&mut self, ray: &Ray) -> Option<(usize, usize)> {
        let r_x = (ray.orig.v[0] * (1000.)) as i64;
        let r_y = (ray.orig.v[1] * (1000.)) as i64;
        let r_z = (ray.orig.v[2] * (1000.)) as i64;

        self.rays.get(&(r_x, r_y, r_z)).cloned()
    }

    pub fn add_ray(&mut self, ray: &Ray) {
        let px = self.get_pixel_for_ray(ray);

        match px {
            Some(pixel) => {
                assert!(!self.checks.contains_key(&pixel));
                self.checks.insert(pixel, 
                    RayDebugCtx {
                        ray: ray.clone(),
                        pixel: pixel,
                        check_tris: Vec::new(),
                        tri_hit: 0,
                        hit_t: 0.
                    });
            },
            None => {}
        }
    }

    pub fn update_ray_triangles(&mut self, ray: &Ray, tris: &Vec<usize>) {
        let px = self.get_pixel_for_ray(ray);

        match px {
            Some(pixel) => {
                let check_ctx = self.checks.get_mut(&pixel).unwrap();

                check_ctx.check_tris.extend(tris);
                check_ctx.check_tris.sort();
                check_ctx.check_tris.dedup();
            }
            None => {
                // println!("Unknown Ray");
            }
        }
    }

    pub fn update_ray_hit(&mut self, ray: &Ray, tri_hit: usize, hit_t: f32) {

        let px = self.get_pixel_for_ray(ray);

        match px {
            Some(pixel) => {
                let check_ctx = self.checks.get_mut(&pixel).unwrap();
                check_ctx.tri_hit = tri_hit;
                check_ctx.hit_t = hit_t;
            }
            None => {
                // println!("Unknown Ray");
            }
        }
    }

    pub fn write_debug_header(&self, writer: &mut dyn io::Write) {
        // Pixel_x;Pixel_y;ray_p;ray_v;tri_hit;hit_t;check_tris;
        write!(writer, "Pixel_x;Pixel_y;ray_p;ray_v;tri_hit;hit_t;check_tris\n").unwrap();
    }

    pub fn write_px_debug_context(&self, pixel: (usize, usize), writer: &mut dyn io::Write) {
        match self.checks.get(&pixel) {
            Some(dr) => {
                write!(writer, "{}\n", dr).unwrap();
            },
            None => {}
        };
    }

    pub fn write_all_debug_context(&self, writer: &mut dyn io::Write) {
        let mut pixels: Vec<&(usize, usize)> = self.checks.keys().collect();

        pixels.sort();

        for px in pixels {
            self.write_px_debug_context(*px, writer);
        }
    }

    fn compare_rays(&self, r1: &Ray, r2: &Ray, diff: f32) -> bool {
        f32::abs(r1.orig.v[0] - r2.orig.v[0]) < diff &&
        f32::abs(r1.orig.v[1] - r2.orig.v[1]) < diff &&
        f32::abs(r1.orig.v[2] - r2.orig.v[2]) < diff &&
        f32::abs(r1.dir.v[0] - r2.dir.v[0]) < diff &&
        f32::abs(r1.dir.v[1] - r2.dir.v[1]) < diff &&
        f32::abs(r1.dir.v[2] - r2.dir.v[2]) < diff
    }

    pub fn compare_to(&self, other: &DebugCtx, writer: &mut dyn io::Write) {
        let mut pixels: Vec<&(usize, usize)> = self.checks.keys().collect();

        pixels.sort();

        let mut err_count = 0;

        for px in pixels {
            let our_ray_ctx = self.checks.get(px).unwrap();

            let their_ray_ctx = match other.checks.get(px) {
                Some(x) => x,
                None => {
                    writeln!(writer, "({},{}): No entry for pixel", px.0, px.1).unwrap();
                    err_count += 1;
                    continue;
                }
            };

            if !self.compare_rays(&our_ray_ctx.ray,
                                  &their_ray_ctx.ray,
                                  0.0001) {
                writeln!(writer, "({},{}): Ray Mismatch {:?} vs {:?}",
                         px.0, px.1,
                         our_ray_ctx.ray,
                         their_ray_ctx.ray).unwrap();
                err_count += 1;
                continue;
            }

            if our_ray_ctx.tri_hit != their_ray_ctx.tri_hit {
                if our_ray_ctx.tri_hit == 0 {
                    if our_ray_ctx.check_tris.contains(&their_ray_ctx.tri_hit) {
                        writeln!(writer, "({},{}): Hit Mismatch {} vs {}. Their hit not in our tri list",
                                 px.0, px.1,
                                 our_ray_ctx.tri_hit,
                                 their_ray_ctx.tri_hit).unwrap();
                    } else {
                        writeln!(writer, "({},{}): Hit Mismatch {} vs {}. Bad hit detection for ray",
                                 px.0, px.1,
                                 our_ray_ctx.tri_hit,
                                 their_ray_ctx.tri_hit).unwrap();
                    }
                } else {
                    if their_ray_ctx.check_tris.contains(&our_ray_ctx.tri_hit) {
                        writeln!(writer, "({},{}): Hit Mismatch {} vs {}. Bad hit detection for ray",
                                 px.0, px.1,
                                 our_ray_ctx.tri_hit,
                                 their_ray_ctx.tri_hit).unwrap();
                    } else {
                        writeln!(writer, "({},{}): Hit Mismatch {} vs {}. Their tri list does not contain our hit",
                                 px.0, px.1,
                                 our_ray_ctx.tri_hit,
                                 their_ray_ctx.tri_hit).unwrap();
                    }
                }
                err_count += 1;
                continue;
            }

            // if f32::abs(our_ray_ctx.hit_t - their_ray_ctx.hit_t) > 0.0001 {
            //     writeln!(writer, "({},{}): Hit times differ {} vs {}",
            //                 px.0, px.1,
            //                 our_ray_ctx.hit_t,
            //                 their_ray_ctx.hit_t).unwrap();
            //     err_count += 1;
            //     continue;
            // }
        }

        writeln!(writer, "Found {} errors", err_count);
    }
}
