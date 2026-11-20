[![license](http://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/udhos/gwob/blob/master/LICENSE)
[![GoDoc](https://godoc.org/github.com/udhos/gwob?status.svg)](http://godoc.org/github.com/udhos/gwob)
[![Go Report Card](https://goreportcard.com/badge/github.com/udhos/gwob)](https://goreportcard.com/report/github.com/udhos/gwob)

# gwob
gwob - Pure Go Golang parser for Wavefront .OBJ 3D geometry file format

# Install

## Install with Go Modules (Go 1.11 or higher)

    git clone https://github.com/udhos/gwob
    cd gwob
    go install

## Install without Go Modules (Go before 1.11)

    go get github.com/udhos/gwob
    cd ~/go/src/github.com/udhos/gwob
    go install github.com/udhos/gwob

# Usage

Import the package in your Go program:

    import "github.com/udhos/gwob"

Example:

    // Error handling omitted for simplicity.

    import "github.com/udhos/gwob"

    options := &gwob.ObjParserOptions{} // parser options

    o, errObj := gwob.NewObjFromFile("gopher.obj", options) // parse/load OBJ

    // Scan OBJ groups
    for _, g := range o.Groups {
        // ...
    }

# Example

Run the example:

    cd example
    go run main.go

You can supply a custom input OBJ by setting the env var INPUT:

    INPUT=gopher.obj go run main.go

If you specify any command line argument, the OBJ will be dumped to stdout:

    go run main.go d

See directory [example](example). 

# Mesh ray picking

For scene preview picking, gwob can build a reusable acceleration structure
(a bounding volume hierarchy) over a parsed OBJ and intersect pick rays with
it. The structure is built once and reused for every query; `Intersect` never
rebuilds it and allocates no memory per query.

    import "math"
    import "github.com/udhos/gwob"

    obj, errObj := gwob.NewObjFromFile("gopher.obj", options)

    // Build once per model. Pass a model (local->world) transform; use
    // IdentityMatrix4() for the raw OBJ coordinates.
    mesh, err := gwob.NewRayMesh(obj, gwob.IdentityMatrix4(), nil)
    if err != nil {
        // One or more triangles were unusable (bad indices, non-finite or
        // degenerate data); mesh still contains every valid triangle.
        // See mesh.InvalidTris and mesh.DegenerateTris.
    }

    ray := gwob.Ray{
        Origin: [3]float64{0, 0, 5},
        Dir:    [3]float64{0, 0, -1}, // need not be normalized
    }

    hit, ok := mesh.Intersect(ray, 0, math.Inf(1))
    if ok {
        hit.T             // ray parameter of the nearest hit
        hit.Point         // world-space hit point
        hit.Bary          // barycentric weights (w0, w1, w2)
        hit.UV            // interpolated texture coordinate (hit.HasUV)
        hit.Normal        // interpolated vertex normal, or geometric normal
        hit.GeometricNormal
        hit.FaceIndex     // original OBJ face ordinal (quads share one)
        hit.TriangleIndex // triangle ordinal inside Obj.Indices
        hit.GroupIndex    // owning Obj.Groups entry
        hit.FrontFace     // respects negative-scale / mirrored transforms
    }

Features:

- Nearest front-or-back intersection over triangles and triangulated quads.
- Affine model transforms including translation, rotation, scale and negative
  (mirroring) scale; rays are transformed into mesh-local space.
- Back-face culling toggle (`RayMeshOptions.CullBackfaces` /
  `mesh.CullBackfaces`) that needs no rebuild.
- Degenerate zero-area triangles, parallel/coplanar rays, out-of-range face
  indices and non-finite vertex or ray coordinates all produce defined
  results: unusable triangles are dropped at build time and counted in
  `InvalidTris` / `DegenerateTris`; bad rays simply miss.
- The original parsed `Obj` data and all existing parser APIs are unchanged.

# Documentation

See the [GoDoc](http://godoc.org/github.com/udhos/gwob) documentation.
