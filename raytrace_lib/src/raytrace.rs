
use core::num;
use core::panic;
use core::slice::SlicePattern;
use std::alloc::alloc;
use std::time;
use std::fs;
use std::io;
use std::io::Result;
use std::thread;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc::{channel, Sender};
use std::collections::{HashMap, VecDeque};
use syscalls::{Sysno, syscall};
use std::f32::consts::{PI, FRAC_PI_4, FRAC_PI_2};
use std::simd::*;
use std::simd::num::SimdFloat;
use std::mem::size_of;
use std::alloc::Allocator;
use log::debug;

use crate::progress;
use crate::bitset::BitSet;
use crate::progress::ProgressStat;
use crate::bumpalloc::BumpAllocator;


#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    v: Simd<f32, 4>
}
pub type Point = Vec3;
pub type Color = Vec3;

pub fn make_vec(v: &[f32; 3]) -> Vec3 {
    Vec3 {
        v: [v[0], v[1], v[2], 0.].into()
    }
}

impl Vec3 {
    #[inline(always)]
    pub fn add(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            v: self.v + other.v
        }
    }

    #[inline(always)]
    pub fn sub(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            v: self.v - other.v
        }
    }

    #[inline(always)]
    pub fn mult(&self, a: f32) -> Vec3 {
        Vec3 {
             v: f32x4::from_array(self.v.to_array().map(|x| x * a))
        }
    }

    #[inline(always)]
    pub fn len2(&self) -> f32 {
        (self.v * self.v).reduce_sum()
    }

    #[inline(always)]
    pub fn len(&self) -> f32 {
        self.len2().sqrt()
    }

    #[inline(always)]
    pub fn dot(&self, other: &Vec3) -> f32 {
        (self.v * other.v).reduce_sum()
    }

    #[inline(always)]
    pub fn cross(&self, other: &Vec3) -> Vec3 {

        let self_1 = simd_swizzle!(self.v, [1, 2, 0, 3]);
        let self_2 = simd_swizzle!(self.v, [2, 0, 1, 3]);
        let other_1 = simd_swizzle!(other.v, [1, 2, 0, 3]);
        let other_2 = simd_swizzle!(other.v, [2, 0, 1, 3]);

        Vec3 {
            v: self_1 * other_2 - self_2 * other_1
        }
        // make_vec(&[self.v[1] * other.v[2] - self.v[2] * other.v[1],
        //            self.v[2] * other.v[0] - self.v[0] * other.v[2],
        //            self.v[0] * other.v[1] - self.v[1] * other.v[0]])
    }

    #[inline(always)]
    pub fn unit(&self) -> Vec3 {
        let res = self.mult(1./self.len());
        res
    }

    pub fn orthogonal(&self) -> Vec3 {
        if self.v[0].abs() > 0.1 {
            make_vec(&[-1.*(self.v[1] + self.v[2]) / self.v[0], 1., 1.]).unit()
        } else if self.v[1].abs() > 0.1 {
            make_vec(&[1., -1.*(self.v[0] + self.v[2]) / self.v[1], 1.]).unit()
        } else if self.v[2].abs() > 0.1 {
            make_vec(&[1., 1., -1.*(self.v[0] + self.v[1]) / self.v[2]]).unit()
        } else {
            self.unit().orthogonal()
        }
    }

    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let n = self.unit();
        let bx = n.orthogonal();
        let by = n.cross(&bx);
        (bx, by, n)
    }

    pub fn change_basis(&self, b: (Vec3, Vec3, Vec3)) -> Vec3 {
        make_vec(&[make_vec(&[b.0.v[0], b.0.v[1], b.0.v[2]]).dot(self),
                   make_vec(&[b.1.v[0], b.1.v[1], b.1.v[2]]).dot(self),
                   make_vec(&[b.2.v[0], b.2.v[1], b.2.v[2]]).dot(self)])
        // Vec3(Vec3(b.0.0, b.0.1, b.0.2).dot(self),
        //      Vec3(b.1.0, b.1.1, b.1.2).dot(self),
        //      Vec3(b.2.0, b.2.1, b.2.2).dot(self))
    }
}

#[inline(never)]
fn vec3_dot_array(a_list: &[Vec3], b_list: &[Vec3]) -> Vec<f32> {
    a_list.iter().zip(b_list).map(|(a, b)| a.dot(b)).collect()
}

fn vec3_dot_array_single(a_list: &[Vec3], b: &Vec3) -> Vec<f32> {
    a_list.iter().map(|a| a.dot(b)).collect()
}

fn vec3_add_array(a_list: &[Vec3], b_list: &[Vec3]) -> Vec<Vec3> {
    a_list.iter().zip(b_list).map(|(a, b)| a.add(b)).collect()
}

fn vec3_add_array_single(a_list: &[Vec3], b: &Vec3) -> Vec<Vec3> {
    a_list.iter().map(|a| a.add(b)).collect()
}

fn vec3_sub_array(a_list: &[Vec3], b_list: &[Vec3]) -> Vec<Vec3> {
    a_list.iter().zip(b_list).map(|(a, b)| a.sub(b)).collect()
}

fn vec3_sub_array_single(a_list: &[Vec3], b: &Vec3) -> Vec<Vec3> {
    a_list.iter().map(|a| a.sub(b)).collect()
}

fn vec3_mult_array(a_list: &[Vec3], b_list: &[f32]) -> Vec<Vec3> {
    a_list.iter().zip(b_list).map(|(a, b)| a.mult(*b)).collect()
}

fn vec3_mult_array_single(a_list: &[Vec3], b: f32) -> Vec<Vec3> {
    a_list.iter().map(|a| a.mult(b)).collect()
}

fn vec3_len2_array(a_list: &[Vec3]) -> Vec<f32> {
    a_list.iter().map(|a| a.len2()).collect()
}

fn vec3_len_array(a_list: &[Vec3]) -> Vec<f32> {
    a_list.iter().map(|a| a.len()).collect()
}


pub fn make_color(color: (u8, u8, u8)) -> Color {
    make_vec(&[(color.0 as f32) / 255.,
               (color.1 as f32) / 255.,
               (color.2 as f32) / 255.])
}

pub fn random_color() -> Color {
    make_color((rand::random::<u8>(),
                rand::random::<u8>(),
                rand::random::<u8>()))
}

fn random_vec() -> Vec3 {
    make_vec(&[rand::random::<f32>() - 0.5,
               rand::random::<f32>() - 0.5,
               rand::random::<f32>() - 0.5]).unit()
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Ray {
    pub orig: Point,
    pub dir: Vec3,
    pub inv_dir: Vec3
}

pub fn make_ray(orig: &Point, dir: &Vec3) -> Ray {
    Ray {
        orig: *orig,
        dir: *dir,
        inv_dir: make_vec(&[1./dir.v[0],
                            1./dir.v[1],
                            1./dir.v[2]])
    }
}

fn ray_intersect_helper(a: &Point, v: &Vec3, b: &Point, u:&Vec3) -> Option<(f32, f32)> {
    let det = u.v[0] * v.v[1] - u.v[1] * v.v[0];
    if det.abs() < 0.0001 {
        return None;
    }

    let dx = b.v[0] - a.v[0];
    let dy = b.v[1] - a.v[1];
    Some((
        (dy * u.v[0] - dx * u.v[1]) / det,
        (dy * v.v[0] - dx * v.v[1]) / det
    ))
}

impl Ray {
    fn at(&self, t: f32) -> Point {
        self.dir.mult(t).add(&self.orig)
    }

    fn intersect(&self, r: &Ray) -> Option<Point> {
        let xy_solution = ray_intersect_helper(&self.orig, &self.dir, &r.orig, &r.dir);
        let (t1, t2) = if xy_solution.is_some() {
            xy_solution.unwrap()
        } else {
            let xz_solution = ray_intersect_helper(
                &make_vec(&[self.orig.v[0], self.orig.v[2], self.orig.v[1]]),
                &make_vec(&[self.dir.v[0], self.dir.v[2], self.dir.v[1]]),
                &make_vec(&[r.orig.v[0], r.orig.v[2], r.orig.v[1]]),
                &make_vec(&[r.dir.v[0], r.dir.v[2], r.dir.v[1]]));
            if xz_solution.is_some() {
                xz_solution.unwrap()
            } else {
                let yz_solution = ray_intersect_helper(
                &make_vec(&[self.orig.v[1], self.orig.v[2], self.orig.v[0]]),
                &make_vec(&[self.dir.v[1], self.dir.v[2], self.dir.v[0]]),
                &make_vec(&[r.orig.v[1], r.orig.v[2], r.orig.v[0]]),
                &make_vec(&[r.dir.v[1], r.dir.v[2], r.dir.v[0]]));

                if yz_solution.is_some() {
                    yz_solution.unwrap()
                } else {
                    return None;
                }
            }
        };

        let p1 = self.at(t1);
        let p2 = r.at(t2);

        let p_diff = p2.sub(&p1);
        if p_diff.len2() < 0.01{
            Some(p1)
        } else {
            None
        }
    }

    fn nearest_point(&self, o: &Point) -> Point {

        let ao = o.sub(&self.orig);
        let ap = self.dir.mult(ao.dot(&self.dir)/self.dir.len2());

        ap.add(&self.orig)
    }
}

fn reflect_ray(orig: &Point, norm: &Vec3, dir: &Vec3, fuzz: f32) -> Ray {

    let ddot = dir.dot(norm).abs();
    let dir_p = norm.mult(ddot);
    let dir_o  = dir.add(&dir_p);

    let reflect = dir_p.add(&dir_o);
    let rand_vec = random_vec().mult(fuzz);

    make_ray(orig, &reflect.add(&rand_vec).unit())
}

fn lambertian_ray(orig: &Point, norm: &Vec3) -> Ray {

    let rand_vec = random_vec();

    make_ray(orig, &norm.add(&rand_vec))
}

fn mix_color(c1: &Color, c2: &Color, a: f32) -> Color {
    c1.mult(1. - a).add(&c2.mult(a))
}

#[derive(Copy, Clone, Debug)]
pub enum SurfaceKind {
    Solid { color: Color },
    Matte { color: Color, alpha: f32},
    Reflective { scattering: f32, color: Color, alpha: f32},
}

#[derive(Debug, PartialEq)]
pub enum CollisionFace {
    Front,
    Back,
    Side,
    Face(usize),
    EdgeFront,
    EdgeBack
}

pub trait Collidable {
    fn intersects(&self, r: &Ray) -> Option<(f32, Point, CollisionFace)>;
    fn normal(&self, p: &Point, f: &CollisionFace) -> Vec3;
    fn getsurface(&self, f: &CollisionFace) -> SurfaceKind;
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct Triangle {
    pub incenter: Point,
    pub norm: Vec3,
    pub sides: [Vec3; 3],
    pub side_lens: [f32; 3],
    // sides: [(Vec3, f64); 3],
    pub corners: [Vec3; 3],
    pub surface: SurfaceKind,
    pub edge_thickness: f32,
}

pub fn make_triangle(points: &[Vec3; 3], surface: &SurfaceKind, edge_thickness: f32) -> Triangle {

    let a = points[0];
    let b = points[points.len()/3];
    let c = points[(points.len()*2)/3];

    let ab = b.sub(&a);
    let ac = c.sub(&a);
    let bc = c.sub(&b);

    let bac_bisect = ac.add(&ab);
    let abc_bisect = bc.add(&ab.mult(-1.));

    let bac_bi_ray = make_ray(&a, &bac_bisect);

    let abc_bi_ray = make_ray(&b, &abc_bisect);

    let incenter = bac_bi_ray.intersect(&abc_bi_ray).unwrap();

    let mut sides = [make_vec(&[0., 0., 0.]); 3];
    let mut side_lens = [0.; 3];
    for idx in 0..points.len() {
        let vedge = points[(idx+1)%points.len()].sub(&points[idx]);
        let po = incenter.sub(&points[idx]);
        let pc = vedge.mult(vedge.dot(&po)/vedge.len2());
        let oc = pc.sub(&po);
        sides[idx] = oc.unit();
        side_lens[idx] = oc.len();
        // sides.push((oc.unit(), oc.len()));
    }

    let norm = sides[0].cross(&sides[1]).unit();

    Triangle {
        incenter: incenter,
        norm: norm,
        // bounding_r2: bounding_len2,
        sides: sides,
        side_lens: side_lens,
        corners: points.clone(),
        surface: *surface,
        edge_thickness: edge_thickness
    }
}

fn populate_triangle_lists(tri_norms: &mut Vec<Vec3, &dyn Allocator>,
                           tri_incenters: &mut Vec<Vec3, &dyn Allocator>,
                           tri_sides: &mut Vec<[Vec3; 3], &dyn Allocator>,
                           tri_sidelens: &mut Vec<[f32; 3], &dyn Allocator>,
                           tri_edgets: &mut Vec<f32, &dyn Allocator>,
                           objs: &BitSet,
                           tris: &[Triangle]) {
    tri_norms.reserve_exact(objs.len());
    tri_incenters.reserve_exact(objs.len()); 
    tri_sides.reserve_exact(objs.len());
    tri_sidelens.reserve_exact(objs.len());
    tri_edgets.reserve_exact(objs.len());

    unsafe {
        tri_norms.set_len(objs.len());
        tri_incenters.set_len(objs.len());
        tri_sides.set_len(objs.len());
        tri_sidelens.set_len(objs.len());
        tri_edgets.set_len(objs.len());
    }

    let mut count = 0;
    for i in objs.iter() {
        let idx = i as usize;
        tri_norms[count] = tris[idx].norm;
        tri_incenters[count] = tris[idx].incenter;
        tri_sides[count] = tris[idx].sides;
        tri_sidelens[count] = tris[idx].side_lens;
        tri_edgets[count] = tris[idx].edge_thickness;

        count += 1;
    }
}
                           

#[inline(never)]
pub fn intersect_triangle_list(r: &Ray, objs: BitSet, tris: &[Triangle], allocator: &dyn Allocator) -> Vec<Option<(f32, Point, CollisionFace, usize)>> {


    let mut tri_norms = Vec::new_in(allocator);
    let mut tri_incenters = Vec::new_in(allocator); 
    let mut tri_sides = Vec::new_in(allocator);
    let mut tri_sidelens = Vec::new_in(allocator);
    let mut tri_edgets = Vec::new_in(allocator);

    populate_triangle_lists(&mut tri_norms, &mut tri_incenters,
                            &mut tri_sides, &mut tri_sidelens,
                            &mut tri_edgets, &objs, tris);
    // let mut count = 0;
    // for i in objs.iter() {
    //     let idx = i as usize;
    //     tri_norms[count] = tris[idx].norm;
    //     tri_incenters[count] = tris[idx].incenter;
    //     tri_sides[count] = tris[idx].sides;
    //     tri_sidelens[count] = tris[idx].side_lens;
    //     tri_edgets[count] = tris[idx].edge_thickness;

    //     count += 1;
    // }

    // for i in objs.iter() {
    //     let idx = i as usize;
    //     tri_norms.push(tris[idx].norm);
    //     tri_incenters.push(tris[idx].incenter);
    //     tri_sides.push(tris[idx].sides);
    //     tri_sidelens.push(tris[idx].side_lens);
    //     tri_edgets.push(tris[idx].edge_thickness);
    // }

    let tnum_list = vec3_dot_array(tri_norms.as_slice(), vec3_sub_array_single(tri_incenters.as_slice(), &r.orig).as_slice());
    let tden_list = vec3_dot_array_single(tri_norms.as_slice(), &r.dir);

    let mut t_list = Vec::with_capacity_in(objs.len(), allocator);
    t_list.extend(tnum_list.iter().zip(tden_list).map(|(tnum, tden)|  tnum / tden));

    let mut p_list = Vec::with_capacity_in(objs.len(), allocator);
    p_list.extend(t_list.iter().map(|t| r.at(*t)));

    let ip_list = vec3_sub_array(p_list.as_slice(), &tri_incenters.as_slice());

    let mut dists_list = Vec::with_capacity_in(objs.len(), allocator);
        // dists_list.extend(ip_list.iter().zip(tri_sides).map(
        //     |(ip, sides)| [ ip.dot(&sides[0]),
        //                     ip.dot(&sides[1]),
        //                     ip.dot(&sides[2])]));
    unsafe {
        dists_list.set_len(objs.len());
    }

    let mut count = 0;
    for i in 0..objs.len() {
        let ip = ip_list[i];
        let sides = tri_sides[i].as_ref();
        dists_list[count] = [ip.dot(&sides[0]),
                             ip.dot(&sides[1]),
                             ip.dot(&sides[2])];
    }


    let mut out_vec = Vec::with_capacity(objs.len());

    debug!("Objs Len: {} tri_incenters: {}", objs.len(), tri_incenters.len());

    unsafe {
        out_vec.set_len(objs.len());
    }

    let mut count = 0;
    for i in 0..objs.len() {

        debug!("Idx : {}", i);
        let sidelens = tri_sidelens[i];
        let dists = dists_list[i];
        let edgethickness = tri_edgets[i];
        let norm = tri_norms[i];
        let p = p_list[i];
        let t = t_list[i];

        let front = r.dir.dot(&norm) > 0.;
        out_vec[count] = if t < 0. {
            None
        } else if dists[0] > sidelens[0] ||
                    dists[1] > sidelens[1] ||
                    dists[2] > sidelens[2] {
            None
        } else if dists[0] > (sidelens[0] * (1. - edgethickness)) ||
                    dists[1] > (sidelens[1] * (1. - edgethickness)) ||
                    dists[2] > (sidelens[2] * (1. - edgethickness)) {
            if front {
                Some((t, p, CollisionFace::EdgeFront, i as usize))
            } else {
                Some((t, p, CollisionFace::EdgeBack, i as usize))
            }
        } else {
            if front {
                Some((t, p, CollisionFace::Front, i as usize))
            } else {
                Some((t, p, CollisionFace::Back, i as usize))
            }
        };

        count += 1;
    }

    out_vec
    
    // dists_list.iter().zip(tri_sidelens)
    //                                 .zip(tri_edgets)
    //                                 .zip(tri_norms)
    //                                 .zip(p_list)
    //                                 .zip(t_list)
    //                                 .zip(objs.iter()).map(
    //     |((((((dists, sidelens), edgethickness), norm), p), t), idx)| {
    //         let front = r.dir.dot(&norm) > 0.;
    //         if t < 0. {
    //             None
    //         } else if dists[0] > sidelens[0] ||
    //                   dists[1] > sidelens[1] ||
    //                   dists[2] > sidelens[2] {
    //             None
    //         } else if dists[0] > (sidelens[0] * (1. - edgethickness)) ||
    //                   dists[1] > (sidelens[1] * (1. - edgethickness)) ||
    //                   dists[2] > (sidelens[2] * (1. - edgethickness)) {
    //             if front {
    //                 Some((t, p, CollisionFace::EdgeFront, idx as usize))
    //             } else {
    //                 Some((t, p, CollisionFace::EdgeBack, idx as usize))
    //             }
    //         } else {
    //             if front {
    //                 Some((t, p, CollisionFace::Front, idx as usize))
    //             } else {
    //                 Some((t, p, CollisionFace::Back, idx as usize))
    //             }
    //         }
    //     }).collect()
    
}

impl Collidable for Triangle {
    fn intersects(&self, r: &Ray) -> Option<(f32, Point, CollisionFace)> {

        let t = self.norm.dot(&self.incenter.sub(&r.orig)) / self.norm.dot(&r.dir);
        if t < 0. {
            return None;
        }
        let p = r.at(t);

        let ip = p.sub(&self.incenter);

        // if ip.len2() > self.bounding_r2 {
        //     return None;
        // }

        let mut hit_edge = false;
        for (side, side_len) in self.sides.iter().zip(self.side_lens) {
            let dist =  ip.dot(&side);
            // println!("{:?} {} {:?} {:?}", dist, side_len, side, ip);
            if dist > side_len {
                return None;
            } else if dist > (side_len * (1. - self.edge_thickness)) {
                hit_edge = true;
            }
        }

        // if ip.dot(&self.side_a) > self.side_a_d ||
        //    ip.dot(&self.side_b) > self.side_b_d ||
        //    ip.dot(&self.side_c) > self.side_c_d {
        //     return None;
        // }

        let side = if hit_edge {
            if r.dir.dot(&self.norm) > 0. {
                CollisionFace::EdgeBack
            } else {
                CollisionFace::EdgeFront
            }
        } else {
            if r.dir.dot(&self.norm) > 0. {
                CollisionFace::Back
            } else {
                CollisionFace::Front
            }
        };

        // let side = if ip.dot(&self.side_a) > self.side_a_d*self.edge ||
        //               ip.dot(&self.side_b) > self.side_b_d*self.edge ||
        //               ip.dot(&self.side_c) > self.side_c_d*self.edge {
        //     CollisionFace::Edge
        // } else {
        //     if r.dir.dot(&self.norm) > 0. {
        //         CollisionFace::Back
        //     } else {
        //         CollisionFace::Front
        //     }
        // };

        Some((t, p, side))
    }

    fn normal(&self, _p: &Point, f: &CollisionFace) -> Vec3 {
        match f {
            CollisionFace::Front => self.norm,
            CollisionFace::Back => self.norm.mult(-1.),
            CollisionFace::EdgeFront => self.norm,
            CollisionFace::EdgeBack => self.norm.mult(-1.),
            _ => panic!("Invalid face for triangle")
        }
    }
    fn getsurface(&self, f: &CollisionFace) -> SurfaceKind {
        match f {
            CollisionFace::EdgeFront => {
                SurfaceKind::Solid { color: make_color((0, 0, 0)) }
            },
            CollisionFace::EdgeBack => {
                SurfaceKind::Solid { color: make_color((0, 0, 0)) }
            },
            _ => self.surface
        }
    }
}


// #[derive(Clone, Debug)]
// pub enum CollisionObject {
//     Sphere(Sphere),
//     Triangle(Triangle),
//     Disk(Disk),
// }

// impl Collidable for CollisionObject {
//     fn intersects(&self, r: &Ray) -> Option<(f64, Point, CollisionFace)> {
//         match self {
//             CollisionObject::Sphere(s) => s.intersects(r),
//             CollisionObject::Triangle(t) => t.intersects(r),
//             CollisionObject::Disk(d) => d.intersects(r),
//         }
//     }

//     fn normal(&self, p: &Point, f: &CollisionFace) -> Vec3 {
//         match self {
//             CollisionObject::Sphere(s) => s.normal(p, f),
//             CollisionObject::Triangle(t) => t.normal(p, f),
//             CollisionObject::Disk(d) => d.normal(p, f),
//         }
//     }

//     fn getsurface(&self, f: &CollisionFace) -> SurfaceKind {
//         match self {
//             CollisionObject::Sphere(s) => s.getsurface(f),
//             CollisionObject::Triangle(t) => t.getsurface(f),
//             CollisionObject::Disk(d) => d.getsurface(f),
//         }
//     }

// }

pub fn make_sphere(orig: &Point, r: f32, lat_lon: (usize, usize), surface: &SurfaceKind, edge_thickness: f32) -> Vec<Triangle> {

    let num_lat = lat_lon.0;
    let num_lon = lat_lon.1;

    assert!(num_lat % 2 == 0);

    let mut tris: Vec<Triangle> = Vec::new();

    for lat_idx in 0..num_lat {
        for lon_idx in 0..num_lon {

            // let top_point = orig.add(&Vec3(r, 0., 0.));
            // let bottom_point = orig.add(&Vec3(-1.*r, 0., 0.));

            let phi1 = (if lat_idx % 2 == 0 {
                (lat_idx as f32) / (num_lat as f32) * PI
            } else {
                ((lat_idx + 1) as f32) / (num_lat as f32) * PI
            } - FRAC_PI_2) * -1.;
            let phi23 = (if lat_idx % 2 == 0 {
                ((lat_idx + 1) as f32) / (num_lat as f32) * PI
            } else {
                (lat_idx as f32) / (num_lat as f32) * PI
            } - FRAC_PI_2) * -1.;

            let smudge = if lat_idx % 2 == 0 { 0. } else { 0.5 };
            let theta1 = ((lon_idx as f32) + smudge) / (num_lon as f32) * 2. * PI;
            let theta2 = ((lon_idx as f32) + 0.5 + smudge) / (num_lon as f32) * 2. * PI;
            let theta3 = ((lon_idx  as f32) - 0.5 + smudge) / (num_lon as f32) * 2. * PI;
            let theta4 = ((lon_idx as f32) + 1.0 + smudge) / (num_lon as f32) * 2. * PI;

            let phi14sin = phi1.sin();
            let phi14cos = phi1.cos();
            let p1 = orig.add(&make_vec(&[r*phi14sin,
                                          r*phi14cos*(theta1.cos()),
                                          r*phi14cos*(theta1.sin())]));

            let phi14sin = phi1.sin();
            let phi14cos = phi1.cos();
            let p4 = orig.add(&make_vec(&[r*phi14sin,
                                          r*phi14cos*(theta4.cos()),
                                          r*phi14cos*(theta4.sin())]));

            let phi23sin = phi23.sin();
            let phi23cos = phi23.cos();
            let p2 = orig.add(&make_vec(&[r*phi23sin,
                                          r*phi23cos*(theta2.cos()),
                                          r*phi23cos*(theta2.sin())]));
            let p3 = orig.add(&make_vec(&[r*phi23sin,
                                          r*phi23cos*(theta3.cos()),
                                          r*phi23cos*(theta3.sin())]));

            println!("{} {}", phi1, phi23);
            println!("{} {} {}", theta1, theta2, theta3);
            println!("{} {} {:?} {:?} {:?}\n", lat_idx, lon_idx, p1, p2, p3);
            tris.push(make_triangle(&[p1, p2, p3],
                                    surface,
                                    edge_thickness));
            if lat_idx != 0 && lat_idx != (num_lat - 1) {
                tris.push(make_triangle(&[p1, p2, p4],
                                        surface,
                                        edge_thickness));
            }
        };
    };

    tris
}

pub fn make_disk(orig: &Point, norm: &Vec3, r: f32, d: f32, num_tris: usize,
                 surface: &SurfaceKind,
                 side_suface: &SurfaceKind, edge_thickness: f32) -> Vec<Triangle> {

    let mut tris: Vec<Triangle> = Vec::new();

    let norm_orth0 = norm.orthogonal().unit().mult(r);
    let norm_orth1 = norm.cross(&norm_orth0).unit().mult(r);
    println!("{:?} {:?} {:?}\n", norm, norm_orth0, norm_orth1);

    // let smudge = 2. * PI / 1000.;
    let smudge = 0.;

    for idx in 0..num_tris {
        let norm_pd = norm.mult(d);
        let norm_md = norm.mult(-1.*d);


        let theta1 = (idx as f32) / (num_tris as f32) * 2. * PI - smudge;
        let theta2 = ((idx as f32) + 1.) / (num_tris as f32) * 2. * PI + smudge;

        let theta3 = ((idx as f32) + 0.5) / (num_tris as f32) * 2. * PI - smudge;
        let theta4 = ((idx as f32) + 1.5) / (num_tris as f32) * 2. * PI + smudge;


        // Top Face
        let p1p = orig.add(&norm_pd);

        let p2p = orig.add(&norm_pd)
                      .add(&norm_orth0.mult(theta1.sin()))
                      .add(&norm_orth1.mult(theta1.cos()));

        let p3p = orig.add(&norm_pd)
                      .add(&norm_orth0.mult(theta2.sin()))
                      .add(&norm_orth1.mult(theta2.cos()));

        tris.push(make_triangle(&[p1p, p2p, p3p],
                                surface, edge_thickness));

        // Bottom Face
        let p1m = orig.add(&norm_md);

        let p2m = orig.add(&norm_md)
                      .add(&norm_orth0.mult(theta3.sin()))
                      .add(&norm_orth1.mult(theta3.cos()));

        let p3m = orig.add(&norm_md)
                      .add(&norm_orth0.mult(theta4.sin()))
                      .add(&norm_orth1.mult(theta4.cos()));

        // tris.push(make_triangle(&[p1m, p2m, p3m],
        //                         surface, edge_thickness));
        
        // Side Face
        tris.push(make_triangle(&[p2p, p3p, p2m],
                                side_suface, edge_thickness));
        tris.push(make_triangle(&[p2m, p3m, p3p],
                                side_suface, edge_thickness));

    };

    tris

}


pub struct LightSource {
    pub orig: Point,
    pub len2: f64
}

// impl LightSource {
//     fn _get_shadow_ray(&self, p: &Point, norm: &Vec3) -> Ray {
        
//         let adj_orig = Vec3(self.orig.0 + rand::random::<f64>() * self.len2,
//                             self.orig.1 + rand::random::<f64>() * self.len2,
//                             self.orig.2 + rand::random::<f64>() * self.len2);
//         let dir = adj_orig.sub(p).unit();
//         let smudge = norm.mult(0.005 * (rand::random::<f64>() + 1.));
//         make_ray(&p.add(&smudge), &dir)
//     }
// }

#[repr(C)]
pub struct BoundingBox {
    pub orig: Point,
    pub len2: f32,
    pub objs: BitSet,
    pub child_boxes: Vec<BoundingBox>,
    pub depth: usize
}

impl Clone for BoundingBox {
    fn clone(&self) -> Self {
        BoundingBox {
            orig: self.orig,
            len2: self.len2,
            objs: self.objs.clone(),
            child_boxes: self.child_boxes.clone(),
            depth: self.depth
        }
    }
}

fn box_contains_point(orig: &Point, len2: f32, p: &Point) -> bool {

    let op = p.sub(orig);

    op.v[0].abs() < len2 &&
    op.v[1].abs() < len2 &&
    op.v[2].abs() < len2
}

fn face_contains_triangle(p: &Point, norm: &Vec3, len2: f32, t: &Triangle) -> bool {
    
    // println!("{:?} {:?} {}", p, norm ,len2);

    let h1 = norm.dot(&p.add(&norm.mult(len2)));
    let h2 = t.norm.dot(&t.incenter);
    let n1 = norm;
    let n2 = t.norm;

    let c1 = (h1 - h2*(n1.dot(&n2)))/(1. - (n1.dot(&n2))*(n1.dot(&n2)));
    let c2 = (h2 - h1*(n1.dot(&n2)))/(1. - (n1.dot(&n2))*(n1.dot(&n2)));

    let line_tmp = make_ray(&n1.mult(c1).add(&n2.mult(c2)),
                            &n1.cross(&n2));

    // Check line collision with box
    let mut tmin = f32::MAX;
    let mut tmax = f32::MAX;
    if norm.v[0] == 0. {
        let t1 = (p.v[0] - len2 - line_tmp.orig.v[0]) * line_tmp.inv_dir.v[0];
        let t2 = (p.v[0] + len2 - line_tmp.orig.v[0]) * line_tmp.inv_dir.v[0];

        // println!("0: t1 t2: {} {}", t1, t2);
        tmin = f32::min(tmin, f32::min(t1, t2));
    };

    if norm.v[1] == 0. {
        let t1 = (p.v[1] - len2 - line_tmp.orig.v[1]) * line_tmp.inv_dir.v[1];
        let t2 = (p.v[1] + len2 - line_tmp.orig.v[1]) * line_tmp.inv_dir.v[1];

        // println!("1: t1 t2: {} {}", t1, t2);
        tmin = f32::min(tmin, f32::min(t1, t2));
    };

    if norm.v[2] == 0. {
        let t1 = (p.v[2] - len2 - line_tmp.orig.v[2]) * line_tmp.inv_dir.v[2];
        let t2 = (p.v[2] + len2 - line_tmp.orig.v[2]) * line_tmp.inv_dir.v[2];

        // println!("2: t1 t2: {} {}", t1, t2);
        tmin = f32::min(tmin, f32::min(t1, t2));
    };

    // println!("tmin: {}", tmin);

    let line = if tmin > 0. {
        line_tmp
    } else {
        // println!("p: {:?}", line_tmp.at(tmin * 2.));
        make_ray(&line_tmp.at(tmin * 2.), &line_tmp.dir)
    };

    // println!("line: {:?}", line);

    // Check line collision with box
    tmin = f32::MIN;
    tmax = f32::MAX;
    if norm.v[0] == 0. {
        let t1 = (p.v[0] - len2 - line.orig.v[0]) * line.inv_dir.v[0];
        let t2 = (p.v[0] + len2 - line.orig.v[0]) * line.inv_dir.v[0];

        // println!("0: t1 t2: {} {}", t1, t2);
        tmin = f32::max(tmin, f32::min(t1, t2));
        tmax = f32::min(tmax, f32::max(t1, t2));
    };

    if norm.v[1] == 0. {
        let t1 = (p.v[1] - len2 - line.orig.v[1]) * line.inv_dir.v[1];
        let t2 = (p.v[1] + len2 - line.orig.v[1]) * line.inv_dir.v[1];

        // println!("1: t1 t2: {} {}", t1, t2);
        tmin = f32::max(tmin, f32::min(t1, t2));
        tmax = f32::min(tmax, f32::max(t1, t2));
    };

    if norm.v[2] == 0. {
        let t1 = (p.v[2] - len2 - line.orig.v[2]) * line.inv_dir.v[2];
        let t2 = (p.v[2] + len2 - line.orig.v[2]) * line.inv_dir.v[2];

        // println!("2: t1 t2: {} {}", tmin, tmax);
        tmin = f32::max(tmin, f32::min(t1, t2));
        tmax = f32::min(tmax, f32::max(t1, t2));
    };

    // println!("tmin tmax: {} {}", tmin, tmax);

    if tmax < tmin {
        return false;
    }

    // Check line collision with triangle
    let t1 = t.corners[0].sub(&line.orig).dot(&line.dir) / line.dir.len2();
    let t2 = t.corners[1].sub(&line.orig).dot(&line.dir) / line.dir.len2();
    let t3 = t.corners[2].sub(&line.orig).dot(&line.dir) / line.dir.len2();
    let p1 = line.at(t1);
    let p2 = line.at(t2);
    let p3 = line.at(t3);

    p1.sub(&t.corners[0]).dot(&p2.sub(&t.corners[1])) < 0. ||
    p1.sub(&t.corners[0]).dot(&p3.sub(&t.corners[2])) < 0. ||
    p2.sub(&t.corners[1]).dot(&p3.sub(&t.corners[2])) < 0.
}

#[cfg(test)]
mod test {
    use crate::raytrace::{Vec3, SurfaceKind, make_vec, make_color, make_triangle, face_contains_triangle};

    #[test]
    fn face_collision() {

        let orig = make_vec(&[2., 2., 2.]);
        let norm = make_vec(&[0., 0., -1.]);
        let len2 = 2.;

        let t = make_triangle(&[make_vec(&[1., 0.4, 0.2]),
                                make_vec(&[1., 0.2, -0.3]),
                                make_vec(&[0.6, 0.6, -0.5])],
                               &SurfaceKind::Solid { color: make_color((0,0,0)) },
                               0.0);
        
// fn face_contains_triangle(p: &Point, norm: &Vec3, len2: f64, t: &Triangle) -> bool {
        assert!(face_contains_triangle(&orig, &norm, len2, &t));

    }
}

pub fn box_contains_polygon(orig: &Point, len2: f32, t: &Triangle) -> bool {

    if box_contains_point(orig, len2, &t.incenter) {
        return true;
    }

    // for corner in &t.corners {
    //     if box_contains_point(orig, len2, &corner) {
    //         return true;
    //     }
    // }

    // Check a ray for each edge of the cube
    let face_norms = [make_vec(&[1., 0., 0.]),
                      make_vec(&[-1., 0., 0.]),
                      make_vec(&[0., 1., 0.]),
                      make_vec(&[0., -1., 0.]),
                      make_vec(&[0., 0., 1.]),
                      make_vec(&[0., 0., -1.])];
    for norm in face_norms {
        if face_contains_triangle(orig, &norm, len2, t) {
            return true;
        }
    }

    false

    // let ab_ray = make_ray(&t.corners[0], &t.corners[1].sub(&t.corners[0]));
    // let ab_p = ab_ray.nearest_point(orig);
    // let ap1 = ab_p.sub(&ab_ray.orig);
    // let ab_v = ap1.dot(&ab_ray.dir);
    // if box_contains_point(orig, len2, &ab_p) &&
    //     ab_v >= 0. && ab_v < ab_ray.dir.len2() {
    //     return true;
    // }

    // let ac_ray = make_ray(&t.corners[0], &t.corners[2].sub(&t.corners[0]));
    // let ac_p = ac_ray.nearest_point(orig);
    // let ap2 = ac_p.sub(&ac_ray.orig);
    // let ac_v = ap2.dot(&ac_ray.dir);
    // if box_contains_point(orig, len2, &ac_p) &&
    //     ac_v >= 0. && ac_v < ac_ray.dir.len2() {
    //     return true;
    // }

    // let bc_ray = make_ray(&t.corners[1], &t.corners[2].sub(&t.corners[1]));
    // let bc_p = bc_ray.nearest_point(orig);
    // let bp = bc_p.sub(&bc_ray.orig);
    // let bc_v = bp.dot(&bc_ray.dir);
    // if box_contains_point(orig, len2, &bc_p) &&
    //     bc_v > 0. && bc_v < bc_ray.dir.len2() {
    //     return true;
    // }

    // let edges: [(usize, usize); 12] = [
    //     (0, 1), (0, 2), (0, 4),
    //     (3, 1), (3, 2), (3, 7),
    //     (5, 1), (5, 4), (5, 7),
    //     (6, 2), (6, 4), (6, 7)
    // ];

    // for edge in edges {
    //     let edge_ray = make_ray(&points[edge.0], &points[edge.1].sub(&points[edge.0]));
    //     match t.intersects(&edge_ray) {
    //         Some((tt, _, __)) => {
    //             if tt >= 0. && tt <= 1. {
    //                 return true;
    //             }
    //         }
    //         None => {}
    //     };
    // }
}

pub fn build_empty_box() -> BoundingBox {
    BoundingBox {
        orig: make_vec(&[0., 0., 0.]),
        len2: 1.,
        objs: BitSet::new(1),
        child_boxes: Vec::new(),
        depth: 0
    }
}

pub fn build_bounding_box(tris: &Vec<Triangle>, orig: &Point, len2: f32, maxdepth: usize, minobjs: usize) -> BoundingBox {
    let mut refvec = BitSet::new(tris.len());
    refvec.extend(0..tris.len());
    build_bounding_box_helper(tris, &refvec, orig, len2, 0, maxdepth, minobjs)
}

fn build_bounding_box_helper(tris: &Vec<Triangle>, objs: &BitSet, orig: &Point, len2: f32, depth: usize, maxdepth: usize, minobjs: usize) -> BoundingBox {

    let mut subobjs = BitSet::new(tris.len());
    for idx in objs.iter() {
        if box_contains_polygon(orig, len2, &tris[idx as usize]) {
            subobjs.insert(idx);
        }
    }

    if subobjs.len() == 0 {
        return BoundingBox {
            orig: *orig,
            len2: len2,
            objs: BitSet::new(1),
            child_boxes: Vec::new(),
            depth: depth
        };
    } else if subobjs.len() < minobjs || depth >= maxdepth {
        return BoundingBox {
            orig: *orig,
            len2: len2,
            objs: subobjs,
            child_boxes: Vec::new(),
            depth: depth
        };
    }

    let mut subboxes: Vec<BoundingBox> = Vec::with_capacity(8);
    let newlen2 = len2 / 2.;
    for i in 0..8 {
        let xoff = if (i & 1) == 0 { -1.*newlen2 } else { newlen2 } ;
        let yoff = if (i & 2) == 0 { -1.*newlen2 } else { newlen2 } ;
        let zoff = if (i & 4) == 0 { -1.*newlen2 } else { newlen2 } ;
        let off_vec = make_vec(&[xoff, yoff, zoff]);
        let bbox = build_bounding_box_helper(tris,
                                             &subobjs,
                                             &orig.add(&off_vec),
                                             newlen2,
                                             depth + 1,
                                             maxdepth,
                                             minobjs);
        subboxes.push(bbox);
    }

    BoundingBox {
        orig: *orig,
        len2: len2,
        objs: BitSet::new(1),
        child_boxes: subboxes,
        depth: depth
    }
}

pub fn build_trivial_bounding_box(tris: &Vec<Triangle>, orig: &Point, len2: f32) -> BoundingBox {

    let mut allobjs = BitSet::new(tris.len());
    allobjs.extend((0..tris.len()).into_iter());
    BoundingBox {
        orig: *orig,
        len2: len2,
        objs: allobjs,
        child_boxes: Vec::new(),
        depth: 0
    }
}

impl BoundingBox {
    
    fn collides_face(&self, r: &Ray, face: usize) -> bool {
        
        let norm = match face {
            0 => make_vec(&[1., 0., 0.]),
            1 => make_vec(&[-1., 0., 0.]),
            2 => make_vec(&[0., 1., 0.]),
            3 => make_vec(&[0., -1., 0.]),
            4 => make_vec(&[0., 0., 1.]),
            5 => make_vec(&[0., 0., -1.]),
            _ => panic!("Invalid face {}", face)
        };

        let ortho_vecs = match face {
            0 => (make_vec(&[0., 1., 0.]), make_vec(&[0., 0., 1.])),
            1 => (make_vec(&[0., 1., 0.]), make_vec(&[0., 0., 1.])),
            2 => (make_vec(&[1., 0., 0.]), make_vec(&[0., 0., 1.])),
            3 => (make_vec(&[1., 0., 0.]), make_vec(&[0., 0., 1.])),
            4 => (make_vec(&[1., 0., 0.]), make_vec(&[0., 1., 0.])),
            5 => (make_vec(&[1., 0., 0.]), make_vec(&[0., 1., 0.])),
            _ => panic!("Invalid face {}", face)
        };

        // t = -1 * (norm * (orig - Rorig)) / (norm * Rdir)
        let surf = self.orig.add(&norm.mult(self.len2));
        let t = norm.dot(&surf.sub(&r.orig)) / norm.dot(&r.dir);
        if t < 0. {
            return false;
        }

        let p = r.at(t);
        let op = p.sub(&self.orig);
        
        op.dot(&ortho_vecs.0).abs() < self.len2 &&
        op.dot(&ortho_vecs.1).abs() < self.len2
    }

    pub fn collides(&self, r: &Ray) -> bool {
    
        let mut tmin = f32::MIN;
        let mut tmax = f32::MAX;
        if r.dir.v[0] != 0. {
            let t1 = (self.orig.v[0] - self.len2 - r.orig.v[0]) * r.inv_dir.v[0];
            let t2 = (self.orig.v[0] + self.len2 - r.orig.v[0]) * r.inv_dir.v[0];

            tmin = f32::max(tmin, f32::min(t1, t2));
            tmax = f32::min(tmax, f32::max(t1, t2));
        }

        if r.dir.v[1] != 0. {
            let t1 = (self.orig.v[1] - self.len2 - r.orig.v[1]) * r.inv_dir.v[1];
            let t2 = (self.orig.v[1] + self.len2 - r.orig.v[1]) * r.inv_dir.v[1];

            tmin = f32::max(tmin, f32::min(t1, t2));
            tmax = f32::min(tmax, f32::max(t1, t2));
        }

        if r.dir.v[2] != 0. {
            let t1 = (self.orig.v[2] - self.len2 - r.orig.v[2]) * r.inv_dir.v[2];
            let t2 = (self.orig.v[2] + self.len2 - r.orig.v[2]) * r.inv_dir.v[2];

            tmin = f32::max(tmin, f32::min(t1, t2));
            tmax = f32::min(tmax, f32::max(t1, t2));
        }

        tmin < tmax
    }

    pub fn get_all_objects_for_ray(&self, tris: &Vec<Triangle>, r: &Ray, path: &mut Vec<BoundingBox>) -> BitSet {

        let mut objmap = BitSet::new(tris.len());

        self.get_all_objects_for_ray_helper(tris, r, &mut objmap, path);
        objmap.update_bitcount();

        objmap
    }

    fn get_all_objects_for_ray_helper(&self, tris: &Vec<Triangle>, r: &Ray, objmap: &mut BitSet, path: &mut Vec<BoundingBox>) {
        if self.objs.len() == 0 &&
           self.child_boxes.len() == 0{
            return;
        }
        if self.collides(r) {
            objmap.orwith(&self.objs);

            if self.child_boxes.len() > 0 {
                for cbox in &self.child_boxes {
                    cbox.get_all_objects_for_ray_helper(tris, r, objmap, path);
                }
            }
        }
    }

    pub fn print_tree(&self) {
        self.print();
        if self.child_boxes.len() > 0 {
            for b in &self.child_boxes {
                b.print_tree();
            }
        }
    }

    pub fn print(&self) {
        println!("BS: {} {:?} {} {} {}",
                 self.depth,
                 self.orig, self.len2,
                 self.objs.len(),
                 self.child_boxes.len()
                 );
    }

    // pub fn find_obj(&self, tris: &Vec<Triangle>, id: usize) -> Vec<&BoundingBox> {
    //     let mut found_list: Vec<&BoundingBox> = Vec::new();
    //     self.find_obj_helper(tris, id, &mut found_list);
    //     found_list
    // }

    // pub fn find_obj_helper(&self, tris: &Vec<Triangle>, id: usize, found_list: &mut Vec<&BoundingBox>) {
    //     if self.objs.len() > 0 {
    //         for obj in &self.objs {
    //             if tris[*obj].getid() == id {
    //                 found_list.push(&self);
    //             }
    //         }
    //     }

    //     if self.child_boxes.len() > 0 {
    //         for b in &self.child_boxes {
    //             b.find_obj_helper(tris, id, found_list);
    //         }
    //     }
    // }

}

pub trait RayCaster: Send + Sync {
    fn project_ray(&self, r: &Ray, s: &Scene, ignore_objid: usize,
                   depth: usize, runtimes: &mut HashMap<String, ProgressStat>,
                   allocator: &dyn Allocator) -> Color;
    fn color_ray(&self, r: &Ray, s: &Scene, objidx: usize,
                 point: &Point, face: &CollisionFace, depth: usize,
                 runtimes: &mut HashMap<String, ProgressStat>,
                 allocator: &dyn Allocator) -> Color;
}

#[derive(Debug)]
#[repr(C)]
struct LinuxTimespec {
    tv_sec: i64,
    tv_nsec: i64
}

pub fn get_thread_time() -> i64 {

    let mut t = LinuxTimespec {
        tv_sec: 0,
        tv_nsec: 0
    };
    match unsafe { syscall!(Sysno::clock_gettime, 3 /*CLOCK_THREAD_CPUTIME_ID*/, &mut t as *mut LinuxTimespec) } {
        Ok(_r) => {
        } 
        Err(err) => {
            panic!("clock_gettime() failed: {}", err)
        }
    };

    t.tv_sec * (1000*1000*1000) + t.tv_nsec
}

#[derive(Copy,Clone)]
pub struct DefaultRayCaster {
}

unsafe impl Send for DefaultRayCaster {}
unsafe impl Sync for DefaultRayCaster {}

impl RayCaster for DefaultRayCaster {
    fn color_ray(&self, r: &Ray, s: &Scene, objidx: usize,
                 point: &Point, face: &CollisionFace, depth: usize,
                 runtimes: &mut HashMap<String, ProgressStat>,
                 allocator: &dyn Allocator) -> Color {

        let shadowed = false;
        // let shadowed = if s.lights.is_some() {
        //     let norm = obj.normal(point, face);
        //     let light_ray = s.lights.as_ref().unwrap().get_shadow_ray(point, &norm);

        //     let mut path: Vec<BoundingBox> = Vec::new();
        //     let mut objs = s.boxes.get_all_objects_for_ray(&light_ray, &mut path);
        //     // objs.extend(s.otherobjs.iter().map(|o| (o.getid(), o)));

        //     let mut found = false;
        //     for (id, search_obj) in objs {
        //         if id != obj.getid() {
        //             if search_obj.intersects(&light_ray).is_some() {
        //                 found = true;
        //                 break;
        //             }
        //         }
        //     }

        //     found
        // } else {
        //     false
        // };

        let black = make_color((0,0,0));

        match s.tris[objidx].getsurface(face) {
            SurfaceKind::Solid {color} => {
                if !shadowed {color} else {black}
            },
            SurfaceKind::Matte {color, alpha} => {
                mix_color(if !shadowed {&color} else {&black},
                        &self.project_ray(&lambertian_ray(point,
                                                    &s.tris[objidx].normal(point, face)),
                                    s,
                                    objidx,
                                    depth - 1,
                                    runtimes,
                                    allocator),
                        alpha)
                
            },
            SurfaceKind::Reflective {scattering, color, alpha}  => {
                mix_color(if !shadowed {&color} else {&black},
                        &self.project_ray(&reflect_ray(point,
                                                    &s.tris[objidx].normal(point, face),
                                                    &r.dir,
                                                    scattering),
                                    s,
                                    objidx,
                                    depth - 1,
                                    runtimes,
                                    allocator),
                        alpha)
            }
        }
    }

    fn project_ray(&self, r: &Ray, s: &Scene, ignore_objid: usize,
                   depth: usize, runstats: &mut HashMap<String, ProgressStat>,
                   allocator: &dyn Allocator) -> Color {

        if depth == 0 {
            return make_color((0, 0, 0));
        }

        let t1 = get_thread_time();

        let blue = make_color((128, 180, 255));

        let mut path: Vec<BoundingBox> = Vec::new();
        let objs = s.boxes.get_all_objects_for_ray(&s.tris, r, &mut path);

        let t2 = get_thread_time();

        let mut objcount = 0;

        // let intersections_checks: Vec<Option<(f32, Vec3, CollisionFace, usize)>> = intersect_triangle_list(r, objs, &s.tris, allocator);

        // let intersections: Vec<&(f32, Vec3, CollisionFace, usize)> = intersections_checks.iter().filter_map(
        //             |x| {
        //                 match x {
        //                     Some(a) => Some(a),
        //                     None => None
        //                 }
        //             }).collect();

        let intersections: Vec<(f32, Point, CollisionFace, usize)> = objs.iter().filter_map(
            |idx| {
                if ignore_objid == idx as usize {
                    None
                } else {
                    objcount += 1;
                    match s.tris[idx as usize].intersects(&r) {
                        Some(p) => Some((p.0, p.1, p.2, idx as usize)),
                        None    => None
                    }
                }
            }).collect();
        
        let t3 = get_thread_time();

        *runstats.entry("BoundingBox".to_string()).or_insert(ProgressStat::Time(time::Duration::from_nanos(0))).as_time_mut() += time::Duration::from_nanos((t2-t1) as u64);
        *runstats.entry("Intersections".to_string()).or_insert(ProgressStat::Time(time::Duration::from_nanos(0))).as_time_mut() += time::Duration::from_nanos((t3-t2) as u64);

        *runstats.entry("Rays".to_string()).or_insert(ProgressStat::Count(0)).as_count_mut() += 1;
        *runstats.entry("TriangleChecks".to_string()).or_insert(ProgressStat::Count(0)).as_count_mut() += objcount;

        if intersections.len() == 0 {
            blue
        } else {
            let (_dist, point, face, objidx) = intersections.iter().fold(&intersections[0],
                |acc, x| {
                    let (dist, _, _, _) = x;
                    let (acc_dist, _, _, _) = acc;
                    if dist < acc_dist { x } else { acc }
                });
                self.color_ray(r, s, *objidx, point, &face, depth, runstats, allocator)

        }
    }

}

#[repr(C)]
pub struct Scene {
    pub tris: Vec<Triangle>,
    pub boxes: BoundingBox,
    // pub otherobjs: Vec<CollisionObject>,
    // pub lights: Option<LightSource>
}

#[derive(Clone, Copy,Debug)]
#[repr(C)]
pub struct Viewport {
    pub width: usize,
    pub height: usize,

    orig: Point,
    cam: Point,
    
    vu: Vec3,
    vv: Vec3,

    maxdepth: usize,
    samples_per_pixel: usize
}

pub fn create_transform(dir_in: &Vec3, d_roll: f32) -> (Vec3, Vec3, Vec3) {

    let dir = dir_in.unit();

    let roll = -1.*(-1. * dir.v[1]).atan2(dir.v[2]);
    let pitch = -1.*dir.v[0].asin();
    let yaw = -1.*d_roll;

    (
        make_vec(&[yaw.cos()*pitch.cos(),
                   yaw.sin()*pitch.cos(),
                   -1.*pitch.sin()]),

        make_vec(&[yaw.cos()*pitch.sin()*roll.sin() - yaw.sin()*roll.cos(),
                   yaw.sin()*pitch.sin()*roll.sin() + yaw.cos()*roll.cos(),
                   pitch.cos()*roll.sin()]),

        make_vec(&[yaw.cos()*pitch.sin()*roll.cos() + yaw.sin()*roll.sin(),
                   yaw.sin()*pitch.sin()*roll.cos() - yaw.cos()*roll.sin(),
                   pitch.cos()*roll.cos()])
    )
}

pub fn create_viewport(px: (u32, u32), size: (f32, f32), pos: &Point, dir: &Vec3, fov: f32, c_roll: f32, maxdepth: usize, samples: usize) -> Viewport {

    let dist = size.0 / (2. * (fov.to_radians() / 2.).tan());

    let rot_basis = create_transform(dir, c_roll);

    let orig = pos.add(&make_vec(&[1.*size.1/2., -1.*size.0/2., 0.]));
    
    let cam_r = make_vec(&[0., 0., dist]).change_basis(rot_basis);
    let cam = pos.sub(&cam_r);

    let vu = make_vec(&[0., size.0, 0.]);
    let vu_r = vu.change_basis(rot_basis);

    let vv = make_vec(&[-1. * size.1, 0., 0.]);
    let vv_r = vv.change_basis(rot_basis);

    Viewport {
        width: px.0 as usize,
        height: px.1 as usize,
        orig: orig,
        cam: cam,
        vu: vu_r,
        vv: vv_r,
        maxdepth: maxdepth,
        samples_per_pixel: samples
    }
}

impl Viewport {

    fn pixel_ray(&self, px: (usize, usize)) -> Ray {

        let px_x = px.0 as f32;
        let px_y = px.1 as f32;

        let vu_delta = self.vu.mult(1. / self.width as f32);
        let vv_delta = self.vv.mult(1. / self.height as f32);

        // let u_off: f32 = 0.5;
        // let v_off: f32 = 0.5;

        let u_off = rand::random::<f32>();
        let v_off = rand::random::<f32>();

        let vu_frac = vu_delta.mult(px_y + u_off);
        let vv_frac = vv_delta.mult(px_x + v_off);

        let px_u = self.orig.add(&vu_frac).add(&vv_frac);

        make_ray(&px_u, &px_u.sub(&self.cam).unit())
    }

    fn walk_ray_set(&self, s: &Scene, rows: Arc<Mutex<VecDeque<(&mut [Color], usize)>>>,
                    t_num: usize, progress_tx: Sender<(usize, usize, usize, usize, HashMap<String, ProgressStat>)>,
                    caster: &dyn RayCaster) {

        let mut rays_count = 0;
        let mut pixels_processed = 0;
        let mut runstats: HashMap<String, ProgressStat> = HashMap::new();
        let bump = BumpAllocator::new((size_of::<Vec3>()*3 +
                                       size_of::<[Vec3; 3]>() + 
                                       size_of::<[f32; 3]>()*2 +
                                       size_of::<f32>()*3) * s.tris.len() * self.maxdepth);
        'threadloop: loop {
            let (data, row) = match rows.lock().unwrap().pop_front() {
                                Some(x) => {
                                    (x.0, x.1)
                                }
                                None => break 'threadloop
                                };
            
            
            let _ = progress_tx.send((t_num, row, 0, 0, runstats.clone()));
            runstats.clear();
            for y in 0..self.width {
                let mut acc = make_vec(&[0., 0., 0.]);
                for _i in 0..self.samples_per_pixel {
                    bump.reset();
                    let ray_color = caster.project_ray(&self.pixel_ray((row,y)), s, usize::MAX, self.maxdepth, &mut runstats, &bump);
                    acc = acc.add(&ray_color);

                    rays_count += 1;
                }
                data[y as usize] = acc.mult(1./(self.samples_per_pixel as f32));
                pixels_processed += 1;
                if rays_count > 10000 {
                    let _ = progress_tx.send((t_num, row, pixels_processed, rays_count, runstats.clone()));
                    rays_count = 0;
                    pixels_processed = 0;
                    runstats.clear();
                }
            }
            let _ = progress_tx.send((t_num, row, pixels_processed, rays_count, runstats.clone()));
            rays_count = 0;
            pixels_processed = 0;
            runstats.clear();
        }
    }

    pub fn walk_rays(&self, s: &Scene, data: & mut[Color], threads: usize,
                     caster: &dyn RayCaster, show_progress: bool) -> progress::ProgressCtx {

        let (progress_tx, progress_rx) = channel();

        let mut progress_io = progress::create_ctx(threads, self.width, self.height, show_progress);


        thread::scope(|sc| {

            let data_parts: Arc<Mutex<VecDeque<(&mut [Color], usize)>>> = 
                Arc::new(Mutex::new(VecDeque::from(
                    data.chunks_mut(self.width).zip(0..self.height).collect::<VecDeque<(&mut [Color], usize)>>())));

            for t in 0..threads {
                let t_progress_tx = progress_tx.clone();
                let t_parts = Arc::clone(&data_parts);
                sc.spawn(move || {
                    self.walk_ray_set(s, t_parts, t, t_progress_tx, caster);
                });
            }

            drop(progress_tx);
            'waitloop: loop {
                match progress_rx.recv() {
                    Ok((t_num, row, y, rays_so_far, runstats)) => {
                        progress_io.update(t_num, row, y, rays_so_far, &runstats);
                    }
                    Err(_x) => {
                        break 'waitloop;
                    }
                };
            }
        });

        progress_io.finish();

        progress_io
    }
}


pub fn write_png(f: fs::File, img_size: (u32, u32), data: &[Color]) -> Result<()> {

    let ref mut w = io::BufWriter::new(f);
    let mut encoder = png::Encoder::new(w, img_size.0, img_size.1);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;

    let mut data_int = Vec::with_capacity(data.len() * 3);
    for c in data {
        data_int.push((c.v[0] * 255.) as u8);
        data_int.push((c.v[1] * 255.) as u8);
        data_int.push((c.v[2] * 255.) as u8);
    }

    writer.write_image_data(&data_int).unwrap();

    Ok(())
}
