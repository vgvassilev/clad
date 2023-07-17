//--------------------------------------------------------------------*- C++ -*-
// clad - The C++ Clang-based Automatic Differentiator
//
// A demo, describing how to use clad in simple AD-aware Path tracer.
//
// author:  Alexander Penev <alexander_penev-at-yahoo.com>
//          Based on smallpt, a Path Tracer by Kevin Beason, 2008
//----------------------------------------------------------------------------//

// To compile the demo please type:
// path/to/clang++ -O3 -Xclang -add-plugin -Xclang clad -Xclang -load -Xclang \
// path/to/libclad.so -I../../include/ -std=c++11 SmallPT.cpp \
// -fopenmp=libiomp5 -o SmallPT
//
// To run the demo please type:
// ./SmallPT 500 && xv image.ppm

// A typical invocation would be:
// ../../../../../obj/Debug+Asserts/bin/clang++ -O3 -Xclang -add-plugin -Xclang clad \
// -Xclang -load -Xclang ../../../../../obj/Debug+Asserts/lib/libclad.dylib \
// -I../../include/ -std=c++11 SmallPT.cpp -fopenmp=libiomp5 -o SmallPT
// ./SmallPT 500 && xv image.ppm

// Necessary for clad to work include
#include "clad/Differentiator/Differentiator.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

//TODO: Try to fix single precision issues
//#define double float

// Test types (default is TEST_TYPE_BY_CLAD)
//#define TEST_TYPE_BY_HAND
#define TEST_TYPE_BY_CLAD
//#define TEST_TYPE_BY_NUM


struct Vec {
  double x, y, z; // position, also color (r,g,b)

  Vec(double x_=0, double y_=0, double z_=0) { x=x_; y=y_; z=z_; }

  Vec operator+(const Vec &b) const { return Vec(x+b.x, y+b.y, z+b.z); }
  Vec operator-(const Vec &b) const { return Vec(x-b.x, y-b.y, z-b.z); }
  Vec operator*(double b) const { return Vec(x*b, y*b, z*b); }
  double operator*(const Vec &b) const { return x*b.x+y*b.y+z*b.z; } // dot
  Vec operator%(const Vec &b) const { return Vec(y*b.z-z*b.y, z*b.x-x*b.z, x*b.y-y*b.x); } // cross
  Vec mult(const Vec &b) const { return Vec(x*b.x, y*b.y, z*b.z); }
  Vec& norm() { return *this = *this * (1/sqrt(x*x+y*y+z*z)); }
};

struct Ray {
  Vec o, d; // Origin and direction
  Ray() : o(), d() {}
  Ray(Vec o_, Vec d_): o(o_), d(d_) {}
};

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

#define inf 1e6
#define eps 1e-6

// Abstract Solid

class Solid {
public:
  Vec e, c; // emission, color
  Refl_t refl; // reflection type (DIFFuse, SPECular, REFRactive)

  Solid() : e(), c(), refl() {}
  Solid(Vec e_, Vec c_, Refl_t refl_): e(e_), c(c_), refl(refl_) {}

  // returns distance, 0 if nohit
  virtual double intersect(const Ray &r) const { return 0; };

  // returns normal vector to surface in point pt
  virtual Vec normal(const Vec &pt) const { return Vec(1,0,0); };
};

// Abstract Implicit Solid

class ImplicitSolid : public Solid {
public:

  ImplicitSolid() {}
  ImplicitSolid(Vec e_, Vec c_, Refl_t refl_): Solid(e_, c_, refl_) {}

  // Return signed distance to nearest point on solid surface
  virtual double distance_func(double x, double y, double z) const {
    return 0;
  }

  // TODO: Remove when forward for virtual diff methods is not need
  virtual double distance_func_darg0(double x, double y, double z) const;
  virtual double distance_func_darg1(double x, double y, double z) const;
  virtual double distance_func_darg2(double x, double y, double z) const;

  // implicit surface intersection
  // returns distance, 0 if nohit
  double intersect(const Ray &r) const override {
    double t=2*eps, t1, f;
    Vec pt;
    do {
      pt=r.o+r.d*t;
      f=fabs(distance_func(pt.x, pt.y, pt.z));
      t1=t;
      t+=f;
      if (f<eps || t==t1) return t;
    } while (t<inf);
    return 0;
  }

  // returns normal vector to surface in point pt
  // by clad
  Vec normal(const Vec &pt) const override {
    // FIXME: Replace the calls to `clad::differentiate` when we fix
    // CladFunction::execute being able to pass the derived class (without
    // it being sliced.
    auto distance_func_dx = clad::differentiate(&ImplicitSolid::distance_func, 0);
    auto distance_func_dy = clad::differentiate(&ImplicitSolid::distance_func, 1);
    auto distance_func_dz = clad::differentiate(&ImplicitSolid::distance_func, 2);
    // FIXME: Uncomment, see above.
    //double Nx = distance_func_dx.execute(*this, pt.x, pt.y, pt.z);
    //double Ny = distance_func_dy.execute(*this, pt.x, pt.y, pt.z);
    //double Nz = distance_func_dz.execute(*this, pt.x, pt.y, pt.z);
    double Nx = distance_func_darg0(pt.x, pt.y, pt.z);
    double Ny = distance_func_darg1(pt.x, pt.y, pt.z);
    double Nz = distance_func_darg2(pt.x, pt.y, pt.z);

    return Vec(Nx, Ny, Nz).norm();
  }
};


// Sphere Solid

#ifdef TEST_TYPE_BY_HAND
// by hand
double sphere_func_dx(double x, double y, double z, const Vec &p, double r) {
  return 2*(x-p.x);
}

double sphere_func_dy(double x, double y, double z, const Vec &p, double r) {
  return 2*(y-p.y);
}

double sphere_func_dz(double x, double y, double z, const Vec &p, double r) {
  return 2*(z-p.z);
}
#endif

double sphere_distance_func(double x, double y, double z, const Vec &p, double r) {
  return sqrt((x-p.x)*(x-p.x) + (y-p.y)*(y-p.y) + (z-p.z)*(z-p.z)) - r;
}

double sphere_implicit_func(double x, double y, double z, const Vec &p, double r) {
  return (x-p.x)*(x-p.x) + (y-p.y)*(y-p.y) + (z-p.z)*(z-p.z) - r*r;
}

//

class Sphere : public ImplicitSolid {
public:
  double r; // radius
  Vec p; // position

  Sphere() : r(), p() {}
  Sphere(double r_, Vec p_, Vec e_, Vec c_, Refl_t refl_):
    r(r_), p(p_), ImplicitSolid(e_, c_, refl_) {
    // Move to Normal when execute can call polymorphic diffs
    auto sphere_func_dx = clad::differentiate(&Sphere::distance_func, 0);
    auto sphere_func_dy = clad::differentiate(&Sphere::distance_func, 1);
    auto sphere_func_dz = clad::differentiate(&Sphere::distance_func, 2);
  }


#ifdef TEST_TYPE_BY_HAND
  // by hand
  Vec normal(const Vec &pt) const override {
    double Nx = sphere_func_dx(pt.x, pt.y, pt.z, p, r);
    double Ny = sphere_func_dy(pt.x, pt.y, pt.z, p, r);
    double Nz = sphere_func_dz(pt.x, pt.y, pt.z, p, r);

    return Vec(Nx, Ny, Nz).norm();
  }
#endif

#ifdef TEST_TYPE_BY_CLAD
  // by clad
  Vec normal(const Vec &pt) const override {
    //auto sphere_func_dx = clad::differentiate(&Sphere::distance_func, 0);
    //auto sphere_func_dy = clad::differentiate(&Sphere::distance_func, 1);
    //auto sphere_func_dz = clad::differentiate(&Sphere::distance_func, 2);

    //double Nx = sphere_func_dx.execute(*this, pt.x, pt.y, pt.z, p, r);
    //double Ny = sphere_func_dy.execute(*this, pt.x, pt.y, pt.z, p, r);
    //double Nz = sphere_func_dz.execute(*this, pt.x, pt.y, pt.z, p, r);

    double Nx = distance_func_darg0(pt.x, pt.y, pt.z);
    double Ny = distance_func_darg0(pt.x, pt.y, pt.z);
    double Nz = distance_func_darg0(pt.x, pt.y, pt.z);

    return Vec(Nx, Ny, Nz).norm();
    //return Vec(Nx, Ny, Nz); // nabla f of signed distance functions is always unit vector
  }
#endif

#ifdef TEST_TYPE_BY_NUM
  // by numeric approximation
  Vec normal(const Vec &pt) const override {
    double f =  sphere_implicit_func(pt.x, pt.y, pt.z, p, r);
    double fx = sphere_implicit_func(pt.x+eps, pt.y, pt.z, p, r);
    double fy = sphere_implicit_func(pt.x, pt.y+eps, pt.z, p, r);
    double fz = sphere_implicit_func(pt.x, pt.y, pt.z+eps, p, r);

    return Vec((fx-f)/eps, (fy-f)/eps, (fz-f)/eps).norm();
  }
#endif

  //TODO: Override distance func method when parent method is virtual
  double distance_func(double x, double y, double z) const override {
    return sqrt((x-p.x)*(x-p.x) + (y-p.y)*(y-p.y) + (z-p.z)*(z-p.z)) - r;
  }
};


// Hyperbolic Solid

#define sin_a 0.965925826289068
#define cos_a 0.258819045102521

#ifdef TEST_TYPE_BY_HAND
// by hand
double h_func_dx(double x, double y, double z, const Vec &p, double r) {
  return (2./3.*cos_a)/pow(((x-p.x)*cos_a+(z-p.z)*sin_a),1./3.) - (2./3.*sin_a)/pow(((z-p.z)*cos_a-(x-p.x)*sin_a),1./3.);
}

double h_func_dy(double x, double y, double z, const Vec &p, double r) {
  return (2./3.)/pow(y-p.y,1./3.);
}

double h_func_dz(double x, double y, double z, const Vec &p, double r) {
  return (2./3.*sin_a)/pow((x-p.x)*cos_a+(z-p.z)*sin_a,1./3.) + (2./3.*cos_a)/pow((z-p.z)*cos_a-(x-p.x)*sin_a,1./3.);
}
#endif

//TODO: Check this distance func. Visualized "octahedron" do not like as octahedron.
double hyperbolic_func(double x, double y, double z, const Vec &p, double r) {
  return pow((x-p.x)*cos_a+(z-p.z)*sin_a, 2./3.) + pow(y-p.y, 2./3.) + pow((x-p.x)*-sin_a+(z-p.z)*cos_a, 2./3.) - pow(r, 2./3.);
}

class HyperbolicSolid : public ImplicitSolid {
public:
  double r; // radius
  Vec p; // position
  HyperbolicSolid() : r(), p() {}
  HyperbolicSolid(double r_, Vec p_, Vec e_, Vec c_, Refl_t refl_)
      : r(r_), p(p_), ImplicitSolid(e_, c_, refl_) {
    // FIXME: Move to Normal when execute can call polymorphic diffs.
    auto hyperbolic_func_dx = clad::differentiate(&HyperbolicSolid::distance_func, 0);
    auto hyperbolic_func_dy = clad::differentiate(&HyperbolicSolid::distance_func, 1);
    auto hyperbolic_func_dz = clad::differentiate(&HyperbolicSolid::distance_func, 2);
  }

#ifdef TEST_TYPE_BY_HAND
  // by hand
  Vec normal(const Vec &pt) const override {
    double Nx = h_func_dx(pt.x, pt.y, pt.z, p, r);
    double Ny = h_func_dy(pt.x, pt.y, pt.z, p, r);
    double Nz = h_func_dz(pt.x, pt.y, pt.z, p, r);

    return Vec(Nx, Ny, Nz).norm();
  }
#endif

#ifdef TEST_TYPE_BY_CLAD
  // by clad
  Vec normal(const Vec &pt) const override {
    //auto hyperbolic_func_dx = clad::differentiate(&HyperbolicSolid::distance_func, 0);
    //auto hyperbolic_func_dy = clad::differentiate(&HyperbolicSolid::distance_func, 1);
    //auto hyperbolic_func_dz = clad::differentiate(&HyperbolicSolid::distance_func, 2);

    //double Nx = hyperbolic_func_dx.execute(*this, pt.x, pt.y, pt.z, p, r);
    //double Ny = hyperbolic_func_dy.execute(*this, pt.x, pt.y, pt.z, p, r);
    //double Nz = hyperbolic_func_dz.execute(*this, pt.x, pt.y, pt.z, p, r);

    double Nx = distance_func_darg0(pt.x, pt.y, pt.z);
    double Ny = distance_func_darg1(pt.x, pt.y, pt.z);
    double Nz = distance_func_darg2(pt.x, pt.y, pt.z);

    // nabla f of signed distance functions is always unit vector, we might
    // need to add support for nabla in clad.
    // If functions are normalized we can skip calling `.norm()`.
    return Vec(Nx, Ny, Nz).norm();
  }
#endif

#ifdef TEST_TYPE_BY_NUM
  // by numeric approximation
  Vec normal(const Vec &pt) const override {
    double f  = hyperbolic_func(pt.x, pt.y, pt.z, p, r);
    double fx = hyperbolic_func(pt.x+eps, pt.y, pt.z, p, r);
    double fy = hyperbolic_func(pt.x, pt.y+eps, pt.z, p, r);
    double fz = hyperbolic_func(pt.x, pt.y, pt.z+eps, p, r);

    return Vec((fx-f)/eps, (fy-f)/eps, (fz-f)/eps).norm();
  }
#endif

  double distance_func(double x, double y, double z) const override {
    return pow((x-p.x)*cos_a+(z-p.z)*sin_a, 2./3.) + pow(y-p.y, 2./3.) + pow((x-p.x)*-sin_a+(z-p.z)*cos_a, 2./3.) - pow(r, 2./3.);
  }
};

// Scene definition (Sphere: radius, position, emission, color, material)
Solid* scene[] = {
  new Sphere(1e5,  Vec(1e5+1, 40.8, 81.6),   Vec(), Vec(.75, .25, .25), DIFF), // Left
  new Sphere(1e5,  Vec(-1e5+99, 40.8, 81.6), Vec(), Vec(.25, .25, .75), DIFF), // Right
  new Sphere(1e5,  Vec(50, 40.8, 1e5),       Vec(), Vec(.75, .75, .75), DIFF), // Back
  new Sphere(1e5,  Vec(50, 40.8, -1e5+170),  Vec(), Vec(),              DIFF), // Front
  new Sphere(1e5,  Vec(50, 1e5, 81.6),       Vec(), Vec(.75, .75, .75), DIFF), // Bottm
  new Sphere(1e5,  Vec(50, -1e5+81.6, 81.6), Vec(), Vec(.75, .75, .75), DIFF), // Top
  new Sphere(16.5, Vec(27, 16.5, 47),        Vec(), Vec(1,1,1)*.999,    SPEC), //Mirror
  new Sphere(16.5, Vec(73, 16.5, 78),        Vec(), Vec(1, 1, 1)*.999,  REFR), // Glass
  new Sphere(600,  Vec(50, 681.6-.27, 81.6), Vec(12,12,12), Vec(),      DIFF)  // Light
};

inline double clamp(double x) {
  return x<0 ? 0 : x>1 ? 1 : x;
}

inline int toInt(double x) {
  return int(pow(clamp(x),1/2.2)*255+.5);
}

inline bool intersect(const Ray &ray, double &t, int &id) {
  double d;

  t = inf;
  for(int i=sizeof(scene)/sizeof(scene[0]); i--; ) {
    if ((d = scene[i]->intersect(ray)) && d<t) { t=d; id=i; }
  }

  return t<inf;
}

Vec radiance(const Ray &ray, int depth, unsigned short *Xi) {
  double t; // distance to intersection
  int id;  // id of intersected object

  Ray r=ray;

  // L0 = Le0 + f0*(L1)
  //    = Le0 + f0*(Le1 + f1*L2)
  //    = Le0 + f0*(Le1 + f1*(Le2 + f2*(L3))
  //    = Le0 + f0*(Le1 + f1*(Le2 + f2*(Le3 + f3*(L4)))
  //    = ...
  //    = Le0 + f0*Le1 + f0*f1*Le2 + f0*f1*f2*Le3 + f0*f1*f2*f3*Le4 + ...
  //
  // So:
  // F = 1
  // while (1) {
  //   L += F*Lei
  //   F *= fi
  // }

  // accumulated color
  Vec cl(0,0,0);
  // accumulated reflectance
  Vec cf(1,1,1);

  while (1) {
    // if miss, return accumulated color (black)
    if (!intersect(r, t, id)) return cl;

    // the hit object
    const Solid &obj = *scene[id];

    // calculate intersection point
    Vec x=r.o+r.d*t;

    // calculate surface normal vector in point x
    Vec n=obj.normal(x);
    Vec nl=n*r.d<0 ? n : n*-1;

    // normal map test
    //return n;

    // object base color
    Vec f=obj.c;
    double p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; // max refl

    // object intersection id map test
    //return obj.c;

    cl = cl + cf.mult(obj.e);

    if (++depth>5 || !p) {
      if (erand48(Xi)<p) f=f*(1/p); else return cl;
    } // R.R.

//    if (depth==0) return obj.c;

    cf = cf.mult(f);

    if (obj.refl == DIFF) { // Ideal DIFFUSE reflection
      double r1=2*M_PI*erand48(Xi), r2=erand48(Xi), r2s=sqrt(r2);
      Vec w=nl, u=((fabs(w.x)>.1 ? Vec(0,1) : Vec(1))%w).norm(), v=w%u;
      Vec d = (u*cos(r1)*r2s+v*sin(r1)*r2s+w*sqrt(1-r2)).norm();

      r = Ray(x, d);
      continue;
    } else if (obj.refl == SPEC) { // Ideal SPECULAR reflection
      //return obj.e + f.mult(radiance(Ray(x,r.d-n*2*(n*r.d)), depth, Xi));
      r = Ray(x, r.d-n*2*(n*r.d));
      continue;
    }

    // Ideal dielectric REFRACTION
    Ray reflRay(x, r.d-n*2*(n*r.d));
    bool into = n*nl>0; // Ray from outside going in?
    double nc=1, nt=1.5, nnt=into ? nc/nt : nt/nc, ddn=r.d*nl, cos2t;

    if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0) { // Total internal reflection
      //return obj.e + f.mult(radiance(reflRay, depth, Xi));
      r = reflRay;
      continue;
    }

    Vec tdir = (r.d*nnt-n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).norm();
    double a=nt-nc, b=nt+nc, R0=a*a/(b*b), c=1-(into?-ddn:tdir*n);
    double Re=R0+(1-R0)*c*c*c*c*c, Tr=1-Re, P=.25+.5*Re, RP=Re/P, TP=Tr/(1-P);

    //return obj.e + f.mult(depth>2 ? // Russian roulette
    //  (erand48(Xi)<P ? radiance(reflRay, depth, Xi)*RP : radiance(Ray(x,tdir), depth, Xi)*TP) :
    //  (radiance(reflRay, depth, Xi)*Re + radiance(Ray(x,tdir), depth, Xi)*Tr) );

    if (erand48(Xi)<P) {
      cf = cf*RP;
      r = reflRay;
    } else {
      cf = cf*TP;
      r = Ray(x,tdir);
    }
    continue;
  }

}

int main(int argc, char *argv[]) {

//  int w=1024, h=768, samps = argc==2 ? atoi(argv[1])/4 : 1; // # samples
  int w=512, h=384, samps = argc==2 ? atoi(argv[1])/4 : 1; // # samples
//  int w=256, h=192, samps = argc==2 ? atoi(argv[1])/4 : 1; // # samples

  int fr = 0;
  Ray cam(Vec(50,52,295.6-fr*10), Vec(0,-0.042612,-1).norm()); // cam pos, dir
  // Different camera setups.
  // Ray cam(Vec(50,52,295.6), Vec(0,-0.042612,-1).norm()); // cam pos, dir
  // Ray cam(Vec(80,52,295.6), Vec(0,-0.042612,-1).norm()); // cam pos, dir
  Vec cx=Vec(w*.5135/h), cy=(cx%cam.d).norm()*.5135, r;
  Vec *frame=new Vec[w*h];

  #pragma omp parallel for schedule(dynamic, 1) private(r)
  for (unsigned short y=0; y<h; y++) { // Loop over image rows
    for (unsigned short x=0, Xi[3]={0,0,(unsigned short)(y*y*y)}; x<w; x++) { // Loop cols
      for (int sy=0, i=(h-y-1)*w+x; sy<2; sy++) { // 2x2 subpixel rows
        for (int sx=0; sx<2; sx++, r=Vec()) {     // 2x2 subpixel cols
          for (int s=0; s<samps; s++) {
            double r1=2*erand48(Xi), dx=r1<1 ? sqrt(r1)-1 : 1-sqrt(2-r1);
            double r2=2*erand48(Xi), dy=r2<1 ? sqrt(r2)-1 : 1-sqrt(2-r2);
            Vec d = cx*(((sx+.5+dx)/2+x)/w-.5)+cy*(((sy+.5+dy)/2+y)/h-.5)+cam.d;
            r = r + radiance(Ray(cam.o+d*140, d.norm()), 0, Xi)*(1./samps);
          } // Camera rays are pushed ^^^^^ forward to start in interior
          frame[i] = frame[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z))*.25;
        }
      }
    }
  }

  // Write image to PPM file.
  char filename[100];
  snprintf(filename, 100, "image-%d.ppm", fr);
  fprintf(stderr, "\rSave %s\n", filename);
  FILE *f = fopen(filename, "wb");
  fprintf(f, "P6\n%d %d\n%d\n", w, h, 255);
  for (int i=0; i<w*h; i++)
    fprintf(f, "%c%c%c", toInt(frame[i].x), toInt(frame[i].y), toInt(frame[i].z));
  fclose(f);
}
