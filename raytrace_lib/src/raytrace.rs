
use core::panic;
use std::time;
use std::fs;
use std::io;
use std::io::Result;
use std::thread;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc::{channel, Sender};
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::f32::consts::{PI, FRAC_PI_2};
use std::simd::*;
use std::simd::num::SimdFloat;
use log::debug;
use ordered_float::OrderedFloat;

use crate::progress;
use crate::progress::ProgressStat;

#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    pub v: Simd<f32, 4>
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
    pub fn mult_per(&self, other: &Vec3) -> Vec3 {
        Vec3 {
             v: self.v * other.v
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
    }
}

#[allow(dead_code)]
#[inline(never)]
fn vec3_dot_array(a_list: &[Vec3], b_list: &[Vec3]) -> Vec<f32> {
    a_list.iter().zip(b_list).map(|(a, b)| a.dot(b)).collect()
}

#[allow(dead_code)]
fn vec3_dot_array_single(a_list: &[Vec3], b: &Vec3) -> Vec<f32> {
    a_list.iter().map(|a| a.dot(b)).collect()
}

#[allow(dead_code)]
fn vec3_add_array(a_list: &[Vec3], b_list: &[Vec3]) -> Vec<Vec3> {
    a_list.iter().zip(b_list).map(|(a, b)| a.add(b)).collect()
}

#[allow(dead_code)]
fn vec3_add_array_single(a_list: &[Vec3], b: &Vec3) -> Vec<Vec3> {
    a_list.iter().map(|a| a.add(b)).collect()
}

#[allow(dead_code)]
fn vec3_sub_array(a_list: &[Vec3], b_list: &[Vec3]) -> Vec<Vec3> {
    a_list.iter().zip(b_list).map(|(a, b)| a.sub(b)).collect()
}

#[allow(dead_code)]
fn vec3_sub_array_single(a_list: &[Vec3], b: &Vec3) -> Vec<Vec3> {
    a_list.iter().map(|a| a.sub(b)).collect()
}

#[allow(dead_code)]
fn vec3_mult_array(a_list: &[Vec3], b_list: &[f32]) -> Vec<Vec3> {
    a_list.iter().zip(b_list).map(|(a, b)| a.mult(*b)).collect()
}

#[allow(dead_code)]
fn vec3_mult_array_single(a_list: &[Vec3], b: f32) -> Vec<Vec3> {
    a_list.iter().map(|a| a.mult(b)).collect()
}

#[allow(dead_code)]
fn vec3_len2_array(a_list: &[Vec3]) -> Vec<f32> {
    a_list.iter().map(|a| a.len2()).collect()
}

#[allow(dead_code)]
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
    let dir_unit = dir.unit();
    Ray {
        orig: *orig,
        dir: dir_unit,
        inv_dir: make_vec(&[1./dir_unit.v[0],
                            1./dir_unit.v[1],
                            1./dir_unit.v[2]])
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

    fn _nearest_point(&self, o: &Point) -> Point {

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

    let reflect_dir = &reflect.add(&rand_vec).unit();

    make_ray(&orig.add(&reflect_dir.mult(0.001)), &reflect.add(&rand_vec).unit())
}

fn lambertian_ray(orig: &Point, norm: &Vec3) -> Ray {

    let rand_vec = random_vec();

    make_ray(&orig.add(&rand_vec.mult(0.001)), &norm.add(&rand_vec))
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
    pub bounding_r2: f32,
    pub sides: [Vec3; 3],
    pub side_lens: [f32; 3],
    pub corners: [Vec3; 3],
    pub surface: SurfaceKind,
    pub edge_thickness: f32
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
    }

    let norm = sides[0].cross(&sides[1]).unit();

    Triangle {
        incenter: incenter,
        norm: norm,
        bounding_r2: points.iter().fold(0.0_f32, | acc, elem | acc.max(elem.sub(&incenter).len2())),
        sides: sides,
        side_lens: side_lens,
        corners: points.clone(),
        surface: *surface,
        edge_thickness: edge_thickness
    }
}

impl Collidable for Triangle {
    fn intersects(&self, r: &Ray) -> Option<(f32, Point, CollisionFace)> {

        let t = self.norm.dot(&self.incenter.sub(&r.orig)) / self.norm.dot(&r.dir);
        if t < 0. {
            return None;
        }
        let p = r.at(t);

        let ip = p.sub(&self.incenter);

        if ip.len2() > self.bounding_r2 {
            return None;
        }

        let mut hit_edge = false;
        for (side, side_len) in self.sides.iter().zip(self.side_lens) {
            let dist =  ip.dot(&side);
            if dist > side_len {
                return None;
            } else if dist > (side_len * (1. - self.edge_thickness)) {
                hit_edge = true;
            }
        }

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


pub fn make_sphere(orig: &Point, r: f32, lat_lon: (usize, usize), surface: &SurfaceKind, edge_thickness: f32) -> Vec<Triangle> {

    let num_lat = lat_lon.0;
    let num_lon = lat_lon.1;

    assert!(num_lat % 2 == 0);

    let mut tris: Vec<Triangle> = Vec::new();

    for lat_idx in 0..num_lat {
        for lon_idx in 0..num_lon {

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

        tris.push(make_triangle(&[p1m, p2m, p3m],
                                surface, edge_thickness));
        
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

#[derive(Clone)]
pub enum BBSubobj {
    Boxes(Vec<BoundingBox>),
    Tris(Vec<usize>)
}

pub struct BoundingBox {
    pub orig: Point,
    pub len2: f32,
    pub objs: BBSubobj,
    pub depth: usize
}

impl Clone for BoundingBox {
    fn clone(&self) -> Self {
        BoundingBox {
            orig: self.orig,
            len2: self.len2,
            objs: self.objs.clone(),
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
    if norm.v[0] == 0. {
        let t1 = (p.v[0] - len2 - line_tmp.orig.v[0]) * line_tmp.inv_dir.v[0];
        let t2 = (p.v[0] + len2 - line_tmp.orig.v[0]) * line_tmp.inv_dir.v[0];

        tmin = f32::min(tmin, f32::min(t1, t2));
    };

    if norm.v[1] == 0. {
        let t1 = (p.v[1] - len2 - line_tmp.orig.v[1]) * line_tmp.inv_dir.v[1];
        let t2 = (p.v[1] + len2 - line_tmp.orig.v[1]) * line_tmp.inv_dir.v[1];

        tmin = f32::min(tmin, f32::min(t1, t2));
    };

    if norm.v[2] == 0. {
        let t1 = (p.v[2] - len2 - line_tmp.orig.v[2]) * line_tmp.inv_dir.v[2];
        let t2 = (p.v[2] + len2 - line_tmp.orig.v[2]) * line_tmp.inv_dir.v[2];

        tmin = f32::min(tmin, f32::min(t1, t2));
    };

    let line = if tmin > 0. {
        line_tmp
    } else {
        make_ray(&line_tmp.at(tmin * 2.), &line_tmp.dir)
    };

    // Check line collision with box
    tmin = f32::MIN;
    let mut tmax = f32::MAX;
    if norm.v[0] == 0. {
        let t1 = (p.v[0] - len2 - line.orig.v[0]) * line.inv_dir.v[0];
        let t2 = (p.v[0] + len2 - line.orig.v[0]) * line.inv_dir.v[0];

        tmin = f32::max(tmin, f32::min(t1, t2));
        tmax = f32::min(tmax, f32::max(t1, t2));
    };

    if norm.v[1] == 0. {
        let t1 = (p.v[1] - len2 - line.orig.v[1]) * line.inv_dir.v[1];
        let t2 = (p.v[1] + len2 - line.orig.v[1]) * line.inv_dir.v[1];

        tmin = f32::max(tmin, f32::min(t1, t2));
        tmax = f32::min(tmax, f32::max(t1, t2));
    };

    if norm.v[2] == 0. {
        let t1 = (p.v[2] - len2 - line.orig.v[2]) * line.inv_dir.v[2];
        let t2 = (p.v[2] + len2 - line.orig.v[2]) * line.inv_dir.v[2];

        tmin = f32::max(tmin, f32::min(t1, t2));
        tmax = f32::min(tmax, f32::max(t1, t2));
    };

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
    use crate::raytrace::{SurfaceKind, make_vec, make_color, make_triangle, face_contains_triangle};

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
        
        assert!(face_contains_triangle(&orig, &norm, len2, &t));

    }
}

pub fn box_contains_polygon(orig: &Point, len2: f32, t: &Triangle) -> bool {

    if box_contains_point(orig, len2, &t.incenter) {
        return true;
    }

    for corner in &t.corners {
        if box_contains_point(orig, len2, &corner) {
            return true;
        }
    }

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
}

pub fn build_empty_box() -> BoundingBox {
    BoundingBox {
        orig: make_vec(&[0., 0., 0.]),
        len2: 1.,
        objs: BBSubobj::Tris(Vec::new()),
        depth: 0
    }
}

pub fn build_bounding_box(tris: &Vec<Triangle>, orig: &Point, len2: f32, maxdepth: usize, minobjs: usize) -> BoundingBox {
    let refvec = (0..tris.len()).collect();
    build_bounding_box_helper(tris, &refvec, orig, len2, 0, maxdepth, minobjs).unwrap()
}

fn build_bounding_box_helper(tris: &Vec<Triangle>, objs: &Vec<usize>, orig: &Point, len2: f32, depth: usize, maxdepth: usize, minobjs: usize) -> Option<BoundingBox> {

    let mut subobjs: Vec<usize> = Vec::new();
    for idx in objs.iter() {
        if box_contains_polygon(orig, len2, &tris[*idx]) {
            subobjs.push(*idx);
        }
    }

    if subobjs.len() == 0 {
        return None;
    } else if subobjs.len() < minobjs || depth >= maxdepth {
        return Some(BoundingBox {
            orig: *orig,
            len2: len2,
            objs: BBSubobj::Tris(subobjs),
            depth: depth
        });
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
        match bbox {
            Some(b) => subboxes.push(b),
            _ => {}
        }
    }

    if subboxes.len() > 0 {
        Some(BoundingBox {
            orig: *orig,
            len2: len2,
            objs: BBSubobj::Boxes(subboxes),
            depth: depth
        })
    } else {
        None
    }
}

pub fn build_trivial_bounding_box(tris: &Vec<Triangle>, orig: &Point, len2: f32) -> BoundingBox {

    let allobjs = (0..tris.len()).collect();
    BoundingBox {
        orig: *orig,
        len2: len2,
        objs: BBSubobj::Tris(allobjs),
        depth: 0
    }
}

impl BoundingBox {
    
    #[inline(never)]
    pub fn collides(&self, r: &Ray) -> Option<(f32, f32)> {
    
        let mut tmin = f32::MIN;
        let mut tmax = f32::MAX;

        let tmp1 = self.orig.sub(&r.orig).mult_per(&r.inv_dir);
        let tmp2 = r.inv_dir.mult(self.len2);

        let t1s = tmp1.sub(&tmp2);
        let t2s = tmp1.add(&tmp2);

        if r.dir.v[0] != 0. {
            // let tmp = (self.orig.v[0] - r.orig.v[0]) * r.inv_dir.v[0];
            // let tmp2 = self.len2 * r.inv_dir.v[0];

            if r.inv_dir.v[0] > 0. {
                tmin = t1s.v[0];
                tmax = t2s.v[0];
            } else {
                tmin = t2s.v[0];
                tmax = t1s.v[0];
            }
        }

        if r.dir.v[1] != 0. {
            // let t1 = (self.orig.v[1] - self.len2 - r.orig.v[1]) * r.inv_dir.v[1];
            // let t2 = (self.orig.v[1] + self.len2 - r.orig.v[1]) * r.inv_dir.v[1];

            if r.inv_dir.v[1] > 0. {
                tmin = f32::max(tmin, t1s.v[1]);
                tmax = f32::min(tmax, t2s.v[1]);
            } else {
                tmin = f32::max(tmin, t2s.v[1]);
                tmax = f32::min(tmax, t1s.v[1]);
            }
        }

        if r.dir.v[2] != 0. {
            // let t1 = (self.orig.v[2] - self.len2 - r.orig.v[2]) * r.inv_dir.v[2];
            // let t2 = (self.orig.v[2] + self.len2 - r.orig.v[2]) * r.inv_dir.v[2];

            if r.inv_dir.v[2] > 0. {
                tmin = f32::max(tmin, t1s.v[2]);
                tmax = f32::min(tmax, t2s.v[2]);
            } else {
                tmin = f32::max(tmin, t2s.v[2]);
                tmax = f32::min(tmax, t1s.v[2]);
            }
        }

        if tmin < tmax {
            Some((tmin, tmax))
        } else {
            None
        }
    }

    #[inline(never)]
    fn get_object_intersection_for_ray(&self, tris: &Vec<Triangle>, r: &Ray) -> Option<(f32, Vec3, CollisionFace, usize)> {

        debug!("{}Bounding box: {}", " ".repeat(self.depth), self.debug_str());
        match &self.objs {
            BBSubobj::Tris(_subts) => {
                debug!("{} Subobjects", " ".repeat(self.depth));
                match self.get_box_min_time_intersection(tris, r) {
                    Some(a) => {
                        debug!("{} Min hit at {} with {}", " ".repeat(self.depth), a.0, a.3);
                        Some(a)
                    },
                    None => {
                        debug!("{} No hits", " ".repeat(self.depth));
                        None
                    }
                }
            },
            BBSubobj::Boxes(subboxes) => {
                debug!("{} Subboxes", " ".repeat(self.depth));
                let mut boxmap: [((f32, f32), Option<&BoundingBox>); 8] = [((f32::MAX, f32::MAX), None); 8];
                subboxes.iter().enumerate().for_each(
                    |(idx, bbox)| {
                        match bbox.collides(r) {
                            Some((tmin, tmax)) => {
                                boxmap[idx] = ((tmin, tmax), Some(&bbox));
                            },
                            _ => {}
                        }
                    });

                // Insertion sort
                for idx in 1..8 {
                    let mut jdx = idx;
                    while jdx > 0 && boxmap[jdx-1].0.0 > boxmap[jdx].0.0 {
                        boxmap.swap(jdx-1, jdx);
                        jdx -= 1;
                    }
                }

                boxmap.iter().fold(((0., 0.), None),
                    |((bboxtmin, bboxtmax), acc): ((f32, f32), Option<(f32, Vec3, CollisionFace, usize)>),
                     ((tmin, tmax), bbox_opt): &((f32, f32), Option<&BoundingBox>) | {
                        match &acc {
                            Some(_) => {
                                debug!("{} Min At {:.2} {:.2}", " ".repeat(self.depth), bboxtmin, bboxtmax);
                            },
                            None => {
                                debug!("{} No hit yet", " ".repeat(self.depth));
                            }
                        };
                        match &bbox_opt {
                            Some(bbox) => {
                                match &acc {
                                    Some(acchit) => {
                                        // if *tmin < (bboxtmax + 0.1) {
                                        if *tmin < bboxtmax {
                                            debug!("{} Checking1 Subbox at {} {} with: {}", " ".repeat(self.depth), tmin, tmax, bbox.debug_str());
                                            let sub = bbox.get_object_intersection_for_ray(tris, r);
                                            match &sub {
                                                Some(subhit) => {
                                                    if subhit.0 < acchit.0 {
                                                        ((*tmin, subhit.0), sub)
                                                    } else {
                                                        ((bboxtmin, bboxtmax), acc)
                                                    }
                                                },
                                                None => {
                                                    ((bboxtmin, bboxtmax), acc)
                                                }
                                            }
                                        } else {
                                            debug!("{} Skipping {:.2} {:.2}", " ".repeat(self.depth), tmin, tmax);
                                            ((bboxtmin, bboxtmax), acc)
                                        }
                                    },
                                    None => {
                                        if *tmin != f32::MAX {
                                            debug!("{} Checking2 Subbox at {} {} with: {}", " ".repeat(self.depth), tmin, tmax, bbox.debug_str());
                                            let sub = bbox.get_object_intersection_for_ray(tris, r);
                                            match &sub {
                                                Some(subhit) => {
                                                    ((*tmin, subhit.0), sub)
                                                },
                                                None => {
                                                    ((0., 0.), None)
                                                }
                                            }
                                        } else {
                                            ((0., 0.), None)
                                        }
                                    }
                                }
                            },
                            None => {
                                ((bboxtmin, bboxtmax), acc)
                            }
                        }
                    }).1
            }
        }
    }

    #[inline(never)]
    fn get_box_min_time_intersection(&self, tris: &Vec<Triangle>, r: &Ray) -> Option<(f32, Vec3, CollisionFace, usize)> {

        match &self.objs {
            BBSubobj::Tris(objtris) => {
                objtris.iter().fold(None,
                    |acc, tnum| {
                        match tris[*tnum].intersects(r) {
                            Some((t, p, face)) => {
                                debug!("{}  Hit {} at {} {:?}", " ".repeat(self.depth), *tnum, t, p);
                                match acc {
                                    Some((tacc, _, _, _)) => {
                                        if t < tacc {
                                            Some((t, p, face, *tnum))
                                        } else {
                                            acc
                                        }
                                    },
                                    None => {
                                        Some((t, p, face, *tnum))
                                    }
                                }
                            },
                            None => {
                                acc
                            }
                        }
                    })
            },
            _ => panic!("Must have objects")
        }

    }

    pub fn get_all_objects_for_ray(&self, tris: &Vec<Triangle>, r: &Ray) -> BTreeMap<OrderedFloat<f32>, Vec<usize>>  {

        let mut objmap = BTreeMap::new();

        self.get_all_objects_for_ray_helper(tris, r, &mut objmap);

        objmap
    }

    fn get_all_objects_for_ray_helper(&self, tris: &Vec<Triangle>, r: &Ray, objmap: &mut BTreeMap<OrderedFloat<f32>, Vec<usize>>) {
        match self.collides(r) {
            Some((tmin, _tmax)) => {
                match &self.objs {
                    BBSubobj::Tris(tris) => {
                        objmap.insert(OrderedFloat(tmin), tris.clone());
                    },
                    BBSubobj::Boxes(c) => {
                        for cbox in c {
                            cbox.get_all_objects_for_ray_helper(tris, r, objmap);
                        }
                    }
                }
            },
            _ => {}
        }
    }

    pub fn print_tree(&self) {
        println!("Bx: {}", self.debug_str());
        match &self.objs {
            BBSubobj::Boxes(c) => {
                for b in c {
                    b.print_tree();
                }
            },
            BBSubobj::Tris(ts) => {
                for t in ts {
                    println!("Obj {}", t);
                }
            }
        }
    }

    pub fn debug_str(&self) -> String {
        format!("{} {:?} {}",
                 self.depth,
                 self.orig, self.len2
                 )
    }
}

// #[derive(Debug)]
// #[repr(C)]
// struct LinuxTimespec {
//     tv_sec: i64,
//     tv_nsec: i64
// }

pub fn get_thread_time() -> i64 {

    0

    // let mut t = LinuxTimespec {
    //     tv_sec: 0,
    //     tv_nsec: 0
    // };
    // match unsafe { syscall!(Sysno::clock_gettime, 3 /*CLOCK_THREAD_CPUTIME_ID*/, &mut t as *mut LinuxTimespec) } {
    //     Ok(_r) => {
    //     } 
    //     Err(err) => {
    //         panic!("clock_gettime() failed: {}", err)
    //     }
    // };

    // t.tv_sec * (1000*1000*1000) + t.tv_nsec
}

pub trait RayCaster: Send + Sync {
    fn walk_rays_internal(&self, v: &Viewport, s: &Scene,
                          data: &mut[Color], threads: usize,
                          progress_tx: Sender<(usize, usize, usize, HashMap<String, ProgressStat>)>);
                        
    fn walk_rays(&self, v: &Viewport, s: &Scene,
                 data: &mut[Color], threads: usize,
                 show_progress: bool) -> progress::ProgressCtx {

        let (progress_tx, progress_rx) = channel();

        let mut progress_io = progress::create_ctx(threads, v.width, v.height, show_progress);

        thread::scope(|sc| {

            let t_progress_tx = progress_tx.clone();
            sc.spawn(move || {
                self.walk_rays_internal(v, s, data, threads, t_progress_tx);
            });

            drop(progress_tx);
            'waitloop: loop {
                match progress_rx.recv() {
                    Ok((t_num, row, y, runstats)) => {
                        progress_io.update(t_num, row, y, &runstats);
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

#[derive(Copy,Clone)]
pub struct DefaultRayCaster {
}

unsafe impl Send for DefaultRayCaster {}
unsafe impl Sync for DefaultRayCaster {}

impl RayCaster for DefaultRayCaster {
    fn walk_rays_internal(&self, v: &Viewport, s: &Scene,
                 data: & mut[Color], threads: usize,
                 progress_tx: Sender<(usize, usize, usize, HashMap<String, ProgressStat>)>) {

        thread::scope(|sc| {

            let data_parts: Arc<Mutex<VecDeque<(&mut [Color], usize)>>> = 
                Arc::new(Mutex::new(VecDeque::from(
                    data.chunks_mut(v.width).zip(0..v.height).collect::<VecDeque<(&mut [Color], usize)>>())));

            for t in 0..threads {
                let t_progress_tx = progress_tx.clone();
                let t_parts = Arc::clone(&data_parts);
                sc.spawn(move || {
                    v.walk_ray_set(s, t_parts, t, t_progress_tx);
                });
            }

            drop(progress_tx);
        });
    }
}


fn color_ray(r: &Ray, s: &Scene, objidx: usize,
             point: &Point, face: &CollisionFace, depth: usize,
             runtimes: &mut HashMap<String, ProgressStat>) -> Color {

    let shadowed = false;
    // let shadowed = if s.lights.is_some() {
    //     let norm = obj.normal(point, face);
    //     let light_ray = s.lights.as_ref().unwrap().get_shadow_ray(point, &norm);

    //     let mut objs = s.boxes.get_all_objects_for_ray(&light_ray);
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
                    &project_ray(&lambertian_ray(point,
                                                 &s.tris[objidx].normal(point, face)),
                                s,
                                depth - 1,
                                runtimes),
                    alpha)
            
        },
        SurfaceKind::Reflective {scattering, color, alpha}  => {
            mix_color(if !shadowed {&color} else {&black},
                    &project_ray(&reflect_ray(point,
                                              &s.tris[objidx].normal(point, face),
                                              &r.dir,
                                              scattering),
                                s,
                                depth - 1,
                                runtimes),
                    alpha)
        }
    }
}

fn project_ray(r: &Ray, s: &Scene, depth: usize, 
               runstats: &mut HashMap<String, ProgressStat>) -> Color {

    debug!("Ray: {:?}", r);

    if depth == 0 {
        return make_color((0, 0, 0));
    }
    let blue = make_color((128, 180, 255));

    let t1 = get_thread_time();
    let hit = s.boxes.get_object_intersection_for_ray(&s.tris, &r);

    let t2 = get_thread_time();

    *runstats.entry("BoundingBox".to_string()).or_insert(ProgressStat::Time(time::Duration::from_nanos(0))).as_time_mut() += time::Duration::from_nanos((t2-t1) as u64);
    *runstats.entry("Intersections".to_string()).or_insert(ProgressStat::Time(time::Duration::from_nanos(0))).as_time_mut() += time::Duration::from_nanos((t2-t1) as u64);

    *runstats.entry("Rays".to_string()).or_insert(ProgressStat::Count(0)).as_count_mut() += 1;
    // *runstats.entry("TriangleChecks".to_string()).or_insert(ProgressStat::Count(0)).as_count_mut() += objcount;

    debug!("{:?}", hit);

    match hit {
        None => blue,
        Some((_t, point, face, objidx)) => {
            color_ray(r, s, objidx, &point, &face, depth, runstats)
        }
    }

}

#[repr(C)]
pub struct Scene {
    pub tris: Vec<Triangle>,
    pub boxes: BoundingBox,
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

    pub maxdepth: usize,
    pub samples_per_pixel: usize
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
        samples_per_pixel: samples,
    }
}

impl Viewport {

    pub fn pixel_ray(&self, px: (usize, usize)) -> Ray {

        let px_x = px.0 as f32;
        let px_y = px.1 as f32;

        let vu_delta = self.vu.mult(1. / self.width as f32);
        let vv_delta = self.vv.mult(1. / self.height as f32);

        let (u_off, v_off) = if self.samples_per_pixel == 1 {
            (0.5, 0.5)
        } else {
            (rand::random::<f32>(), rand::random::<f32>())
        };

        let vu_frac = vu_delta.mult(px_y + u_off);
        let vv_frac = vv_delta.mult(px_x + v_off);

        let px_u = self.orig.add(&vu_frac).add(&vv_frac);

        make_ray(&px_u, &px_u.sub(&self.cam).unit())
    }

    fn walk_ray_set(&self, s: &Scene, rows: Arc<Mutex<VecDeque<(&mut [Color], usize)>>>,
                    t_num: usize, progress_tx: Sender<(usize, usize, usize, HashMap<String, ProgressStat>)>) {

        let mut rays_count = 0;
        let mut pixels_processed = 0;
        let mut runstats: HashMap<String, ProgressStat> = HashMap::new();
        'threadloop: loop {
            let (data, row) = match rows.lock().unwrap().pop_front() {
                                Some(x) => {
                                    (x.0, x.1)
                                }
                                None => break 'threadloop
                                };
            
            
            let _ = progress_tx.send((t_num, row, 0, runstats.clone()));
            runstats.clear();
            for y in 0..self.width {
                let mut acc = make_vec(&[0., 0., 0.]);
                for _i in 0..self.samples_per_pixel {
                    let ray_color = project_ray(&self.pixel_ray((row,y)), s, self.maxdepth, &mut runstats);
                    acc = acc.add(&ray_color);

                    rays_count += 1;
                }
                data[y as usize] = acc.mult(1./(self.samples_per_pixel as f32));
                pixels_processed += 1;
                if rays_count > 10000 {
                    let _ = progress_tx.send((t_num, row, pixels_processed, runstats.clone()));
                    rays_count = 0;
                    pixels_processed = 0;
                    runstats.clear();
                }
            }
            let _ = progress_tx.send((t_num, row, pixels_processed, runstats.clone()));
            rays_count = 0;
            pixels_processed = 0;
            runstats.clear();
        }
    }

    pub fn walk_one_ray(&self, s: &Scene, data: & mut[Color], px: (usize, usize)) -> progress::ProgressCtx {

        let mut progress_io = progress::create_ctx(1, self.width, self.height, false);

        let mut runstats: HashMap<String, ProgressStat> = HashMap::new();

        let ray_color = project_ray(&self.pixel_ray((px.1,px.0)), s, self.maxdepth, &mut runstats);

        data[0] = ray_color;

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
