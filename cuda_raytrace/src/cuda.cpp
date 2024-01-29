
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <png.h>
#include <assert.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <cstring>

struct Color {
    uint8_t r,g,b;
};

enum SurfaceKind {
    SOLID
};

struct Surface {
    SurfaceKind kind;
    union {
        struct {
            Color c;
        } solid;
    };
};

struct Vector;

struct Point {
    float x,y,z;

    Vector add(const Vector& b) const;
    Vector sub(const Vector& b) const;

    void print() const {
        printf("(%.3f %.3f %.3f)", x, y, z);
    }
};

struct Vector {
    float x,y,z;

    Vector() :
        x(0),
        y(0),
        z(0)
    {};

    Vector(const float x, const float y, const float z) :
        x(x),
        y(y),
        z(z)
    {};

    Vector(const Point& a, const Point& b) {
        this->x = b.x - a.x;
        this->y = b.y - a.y;
        this->z = b.z - a.z;
    };

    Vector(const Point& a) :
        x(a.x),
        y(a.y),
        z(a.z)
    {};

    Point to_point(void) const {
        return Point {
            .x = this->x,
            .y = this->y,
            .z = this->z,
        };
    };

    Point to_point(const Point& a) const {
        return Point {
            .x = a.x + this->x,
            .y = a.y + this->y,
            .z = a.z + this->z,
        };
    };

    Vector add(const Vector& o) const {
        return Vector(this->x + o.x,
                      this->y + o.y,
                      this->z + o.z);
    };

    Vector sub(const Vector& o) const {
        return Vector(this->x - o.x,
                      this->y - o.y,
                      this->z - o.z);
    };

    Vector mult(const float t) const {
        return Vector(this->x * t,
                      this->y * t,
                      this->z * t);
    }

    float dot(const Vector& o) const {
        return this->x * o.x +
               this->y * o.y +
               this->z * o.z;
    };

    float len2() const {
        return this->x*this->x + this->y*this->y + this->z*this->z;
    };

    float len() const {
        return sqrt(this->len2());
    };

    Vector norm() const {
        return this->mult(1./this->len());
    };

    Vector cross(const Vector& o) const {
        return Vector(this->y * o.z - this->z * o.y,
                      this->z * o.x - this->x * o.z,
                      this->x * o.y - this->y * o.x);
    };

    void print() const {
        printf("(%.3f %.3f %.3f)", x, y, z);
    }
};

Vector Point::add(const Vector& b) const {
    return Vector(*this).add(b);
}

Vector Point::sub(const Vector& b) const {
    return Vector(*this).sub(b);
}

struct Ray;

struct Plane {
    Point a;
    Vector n;

    float intersect_ray(const Ray& r) const;
};

struct Ray {
    Point a;
    Vector u;

    Ray(const Point& a, const Vector& u) :
        a(a),
        u(u)
    {};

    Ray(const Point& a, const Point& b) :
        a(a),
        u(Vector(a, b))
    {};

    Point at(const float t) const {
        Vector va(a);
        Vector ut = u.mult(t);
        return va.add(ut).to_point();
    };

    void print() const {
        printf("{P: ");
        a.print();
        printf(" u: ");
        u.print();
        printf("}");
    }
};

float Plane::intersect_ray(const Ray& r) const {
    Vector op = Vector(this->a.sub(r.a));
    return op.dot(this->n) / r.u.dot(this->n);
}

struct Triangle;

struct Hit {
    bool valid;
    float t;
    const Triangle& tri;

    static bool minhit(const Hit& a, const Hit& b) {
        if (a.valid && b.valid) {
            return a.t < b.t;
        } else if (b.valid) {
            return false;
        } else {
            return true;
        }
    };
};

struct Triangle {
    const Point a,b,c;
    const Vector ab, bc, ca;
    const Vector norm;
    const Surface s;

    Triangle(const Point points[3], const Surface s) :
        a(points[0]),
        b(points[1]),
        c(points[2]),
        ab(points[0], points[1]),
        bc(points[1], points[2]),
        ca(points[2], points[0]),
        norm(ab.cross(bc).norm()),
        s(s)
    {};

    Hit check_intersection(const Ray& r) const {
        Plane plane = Plane {
            .a = this->a,
            .n = this->norm
        };
        const float t = plane.intersect_ray(r);

        if (t < 0.) {
            return Hit {
                .valid = false,
                .t = t,
                .tri = *this
            };
        }

        const Point p = r.at(t);

        Vector ap = Vector(a, p);
        Vector bp = Vector(b, p);
        Vector cp = Vector(c, p);

        if (this->ab.cross(ap).dot(this->norm) >= 0 &&
            this->bc.cross(bp).dot(this->norm) >= 0 &&
            this->ca.cross(cp).dot(this->norm) >= 0) {

            return Hit {
                .valid = true,
                .t = t,
                .tri = *this
            };
        } else {
            return Hit {
                .valid = false,
                .t = t,
                .tri = *this
            };
        }
    };
};

struct Viewport {
    int width, height;
    Point orig, cam;
    Vector vu, vv;
    int spp;

    Viewport(int width, int height, int spp) :
        width(width),
        height(height),
        spp(spp) {

        float aspect = (float)height / (float)width;
        
        this->vu = Vector(0, 1., 0.);
        this->vv = Vector(-1 * aspect, 0, 0);

        this->orig = Point {
            .x = aspect / 2,
            .y = -0.5,
            .z = 0
        };

        this->cam = Point {0, 0, -1};
    };

    Ray getRay(int r, int c) const {
        Vector vv_r = vv.mult(((r + 0.5) / this->height));
        Vector vu_c = vu.mult(((c + 0.5) / this->width));
        Point r_a = this->orig.add(vv_r).add(vu_c).to_point();
        return Ray(r_a, Vector(this->cam, r_a));
    }
};

Color get_color(const Hit h) {
    if (h.valid) {
        const Surface& s = h.tri.s;
        switch (s.kind) {
            case SOLID:
                return s.solid.c;
            default:
                assert(false);
        }
    } else {
        return Color {
                .r = 128,
                .g = 200,
                .b = 255
               };
    }
}

Hit project_ray(const Ray& r, std::vector<Triangle>& tris) {
    std::vector<Hit> hits;
    for (const Triangle& tri : tris) {
        hits.push_back(tri.check_intersection(r));
    }
    return *std::min_element(hits.begin(), hits.end(), Hit::minhit);
}

void trace_rays(const Viewport& v, std::vector<Triangle>& tris, Color** rows) {
    for (int r = 0; r < v.height; r++) {
        printf("%d/%d\n", r, v.height);
        rows[r] = new Color[v.width];
        for (int c = 0; c < v.width; c++) {
            const Ray ray = v.getRay(r, c);
            const Hit h = project_ray(ray, tris);

            rows[r][c] = get_color(h);
        }
    }
}

void write_png(const char* fname, uint8_t** data, int width, int height) {

    FILE* fp = fopen(fname, "wb");
    assert(fp != nullptr);

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    assert(png != nullptr);

    png_infop info = png_create_info_struct(png);
    assert(info != nullptr);

    png_init_io(png, fp);

    png_set_IHDR(
        png, info, width, height,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );

    png_write_info(png, info);
    png_write_image(png, data);
    png_write_end(png, nullptr);

    fclose(fp);

    png_destroy_write_struct(&png, &info);
}



std::vector<Triangle> parse_obj(const char* fname) {
    std::ifstream objfile;
    objfile.open(fname);

    std::vector<Point> vertices;
    std::vector<std::tuple<int, int ,int>> faces;

    for (std::string line; std::getline(objfile, line);) {
        if (strncmp(line.c_str(), "v ", 2) == 0) {
            float a,b,c;
            sscanf(line.c_str(), "v %f %f %f", &a, &b, &c);
            vertices.push_back({a, b, c});
        }
        if (strncmp(line.c_str(), "f ", 2) == 0) {
            int a,an;
            int b,bn;
            int c,cn;

            sscanf(line.c_str(), "f %d//%d %d//%d %d//%d",
                                 &a, &an, &b, &bn, &c, &cn);
            faces.push_back(std::make_tuple(a, b, c));
        }
    }

    std::vector<Triangle> tris;

    for (auto face : faces) {

        Point points[3] = {
            vertices[std::get<0>(face)-1],
            vertices[std::get<1>(face)-1],
            vertices[std::get<2>(face)-1],
        };

        Vector adj(0, 0, 5);
        Point points_adj[3] = {
            points[0].add(adj).to_point(),
            points[1].add(adj).to_point(),
            points[2].add(adj).to_point()
        };

        Surface s = {
            .kind = SOLID,
            .solid = {{255, 0, 0}}
        };
        tris.push_back(Triangle(points_adj, s));
    }

    return tris;
}

int main(int argc, char** argv) {
    
    const int width = 640;
    const int height = 480;

    Color** colors = new Color*[height];
    Viewport v = Viewport(width, height, 1);


    auto tris = parse_obj("teapot_tri.obj");
    // printf("Found %ld objs\n", tris.size());

    // std::vector<Triangle> tris;
    // Point points[3] = {
    //     {-0.5, -1, 5},
    //     {-0.5, 1, 5},
    //     {1, 0, 5},
    // };

    // Surface s = {
    //     .kind = SOLID,
    //     .solid = {{255, 0, 0}}
    // };
    // tris.push_back(Triangle(points, s));

    // Point points2[3] = {
    //     {0.5, 0.2, 4},
    //     {-.5, 1, 4},
    //     {1, 2, 5},
    // };

    // tris.push_back(Triangle(points2, s));

    // Ray r(Point {0, 0, 0}, Point {0, 0, 1});
    // Hit h = tris[0].check_intersection(r);
    // printf("Hit %d %.3f", h.valid, h.t);
    // return 0;

    trace_rays(v, tris, colors);
    
    uint8_t** rbg_data = new uint8_t*[height];

    for (int r = 0; r < height; r++) {
        rbg_data[r] = new uint8_t[width*3];
        for (int c = 0; c < width; c++) {
            rbg_data[r][c*3] = colors[r][c].r;
            rbg_data[r][c*3+1] = colors[r][c].g;
            rbg_data[r][c*3+2] = colors[r][c].b;
        }
    }

    write_png("cpp_test.png", rbg_data, width, height);

    return 0;
}