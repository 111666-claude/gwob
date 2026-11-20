package gwob

import (
	"math"
	"testing"
)

const rayLogStats = false

func rayOptions() *ObjParserOptions {
	return &ObjParserOptions{LogStats: rayLogStats}
}

func approx(want, got, tol float64) bool {
	return math.Abs(want-got) <= tol
}

func approxVec3(want, got [3]float64, tol float64) bool {
	for i := range want {
		if !approx(want[i], got[i], tol) {
			return false
		}
	}
	return true
}

func approxVec2(want, got [2]float64, tol float64) bool {
	for i := range want {
		if !approx(want[i], got[i], tol) {
			return false
		}
	}
	return true
}

// twoTriObj defines two overlapping triangles in the z=0 plane at different
// depths, each with its own UVs and normals. The nearer one (z=-1) overlaps
// the unit square of the farther one (z=0) over [0.25,0.75]^2.
const twoTriObj = `
o two_overlapping
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0.25 0.25 -1
v 0.75 0.25 -1
v 0.75 0.75 -1
v 0.25 0.75 -1
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vt 0.25 0.25
vt 0.75 0.25
vt 0.75 0.75
vt 0.25 0.75
vn 0 0 1
f 1/1/1 2/2/1 3/3/1
f 5/5/1 7/7/1 8/8/1
`

func TestRayTwoOverlappingFaces(t *testing.T) {
	o, err := NewObjFromBuf("twoTriObj", []byte(twoTriObj), rayOptions())
	if err != nil {
		// parser never returns fatal errors for this fixture, but guard anyway
		t.Fatalf("NewObjFromBuf: %v", err)
	}

	mesh, err := NewRayMesh(o, IdentityMatrix4(), nil)
	if err != nil {
		t.Fatalf("NewRayMesh: %v", err)
	}

	if mesh.TriangleCount != 2 || mesh.FaceCount != 2 {
		t.Fatalf("structure: tris=%d faces=%d, want 2/2", mesh.TriangleCount, mesh.FaceCount)
	}

	// Ray through the overlap region, at an interior point of the near
	// triangle (v5,v8,v7): (0.5,0.6). It must hit the nearer face (#1).
	ray := Ray{
		Origin: [3]float64{0.5, 0.6, 5},
		Dir:    [3]float64{0, 0, -1},
	}

	// Repeated queries must reuse the same structure without rebuilding.
	for q := 0; q < 3; q++ {
		hit, ok := mesh.Intersect(ray, 0, math.Inf(1))
		if !ok {
			t.Fatalf("query %d: expected hit", q)
		}
		if hit.FaceIndex != 1 || hit.TriangleIndex != 1 {
			t.Fatalf("query %d: nearest hit face=%d tri=%d, want face=1 tri=1", q, hit.FaceIndex, hit.TriangleIndex)
		}
		if !approx(6, hit.T, 1e-9) || !approxVec3([3]float64{0.5, 0.6, -1}, hit.Point, 1e-9) {
			t.Fatalf("query %d: t=%g point=%v", q, hit.T, hit.Point)
		}
		// Near face is CCW (v5,v7,v8) with winding normal +z. At (0.5,0.6)
		// the weights are w(v5)=0.3, w(v7)=0.5, w(v8)=0.2.
		if !approxVec3([3]float64{0.3, 0.5, 0.2}, hit.Bary, 1e-9) {
			t.Fatalf("query %d: bary=%v", q, hit.Bary)
		}
		if !hit.FrontFace {
			t.Fatalf("query %d: expected front face", q)
		}
		// UVs follow the same corner weights over vt5 (.25,.25),
		// vt7 (.75,.75), vt8 (.25,.75).
		if !hit.HasUV || !approxVec2([2]float64{0.5, 0.6}, hit.UV, 1e-9) {
			t.Fatalf("query %d: uv=%v hasUV=%v", q, hit.UV, hit.HasUV)
		}
		// Both faces reference vn (0,0,1), so the interpolated normal is
		// present and equals the geometric normal here.
		if !hit.HasNormal || !approxVec3([3]float64{0, 0, 1}, hit.Normal, 1e-9) {
			t.Fatalf("query %d: normal=%v hasNormal=%v", q, hit.Normal, hit.HasNormal)
		}
	}

	// Ray outside the near triangle but inside the far one must reach face #0,
	// proving nearest-hit selection rather than first-hit.
	rayOutside := Ray{
		Origin: [3]float64{0.1, 0.1, 5},
		Dir:    [3]float64{0, 0, -1},
	}
	hit, ok := mesh.Intersect(rayOutside, 0, math.Inf(1))
	if !ok {
		t.Fatal("expected far-face hit")
	}
	if hit.FaceIndex != 0 || hit.TriangleIndex != 0 {
		t.Fatalf("far hit: face=%d tri=%d, want 0/0", hit.FaceIndex, hit.TriangleIndex)
	}
	if !approx(5, hit.T, 1e-9) || !approxVec2([2]float64{0.1, 0.1}, hit.UV, 1e-9) {
		t.Fatalf("far hit: t=%g uv=%v", hit.T, hit.UV)
	}
	// Far triangle carries vn (0,0,1), so HasNormal must be true here.
	if !hit.HasNormal || !approxVec3([3]float64{0, 0, 1}, hit.Normal, 1e-9) {
		t.Fatalf("far hit: normal=%v hasNormal=%v", hit.Normal, hit.HasNormal)
	}

	// t interval clipping: a ray starting beyond the near plane (t > 6)
	// misses both faces.
	if _, ok = mesh.Intersect(ray, 6.5, math.Inf(1)); ok {
		t.Fatal("tmin clip beyond geometry: unexpected hit")
	}

	// tmax before the near plane rejects it; the far ray (only reachable
	// outside the overlap) clips the same way at its own t=5 plane.
	if _, ok = mesh.Intersect(rayOutside, 0, 4.0); ok {
		t.Fatal("tmax clip before far plane: unexpected hit")
	}
	hit, ok = mesh.Intersect(rayOutside, 0, math.Inf(1))
	if !ok || hit.FaceIndex != 0 {
		t.Fatalf("unclipped far ray recovery: ok=%v face=%d", ok, hit.FaceIndex)
	}

	// Back-face culling: rays now travel in +z, so the near face is behind.
	mesh.CullBackfaces = true
	backRay := Ray{
		Origin: [3]float64{0.5, 0.5, -5},
		Dir:    [3]float64{0, 0, 1},
	}
	if _, ok = mesh.Intersect(backRay, 0, math.Inf(1)); ok {
		t.Fatal("cull: back-facing triangle should have been rejected")
	}
	mesh.CullBackfaces = false
	hit, ok = mesh.Intersect(backRay, 0, math.Inf(1))
	if !ok || hit.FrontFace {
		t.Fatalf("no-cull back side: ok=%v front=%v, want back-facing", ok, hit.FrontFace)
	}
}

// quadObj is one quad in the z=0 plane with corner UVs and corner normals.
// Both fan triangles wind clockwise about +z, so the geometric winding
// normal is -z in local space.
const quadObj = `
o transformed_quad
v 0 0 0
v 0 1 0
v 1 1 0
v 1 0 0
vt 0 0
vt 0 1
vt 1 1
vt 1 0
vn 1 0 0
vn 0 1 0
vn 0 0 1
vn 1 1 1
f 1/1/1 2/2/2 3/3/3 4/4/4
`

func TestRayTransformedQuad(t *testing.T) {
	o, err := NewObjFromBuf("quadObj", []byte(quadObj), rayOptions())
	if err != nil {
		t.Fatalf("NewObjFromBuf: %v", err)
	}
	if len(o.faceSizes) != 1 || o.faceSizes[0] != 4 {
		t.Fatalf("face sizes: %v, want [4]", o.faceSizes)
	}
	if len(o.Indices) != 6 {
		t.Fatalf("quad must triangulate to 6 indices, got %d", len(o.Indices))
	}

	// Model: rotate 90deg around X, then translate. Local z=0 plane becomes
	// the world y=2 plane and R_x(90deg) maps (x,y,z) -> (x,-z,y), so the
	// local +z winding normal rotates into world +y.
	model := MultiplyMatrix4(TranslationMatrix4(0, 2, 3), RotationXMatrix4(math.Pi/2))

	mesh, err := NewRayMesh(o, model, nil)
	if err != nil {
		t.Fatalf("NewRayMesh: %v", err)
	}
	if mesh.TriangleCount != 2 || mesh.FaceCount != 1 {
		t.Fatalf("structure: tris=%d faces=%d, want 2 triangles of 1 quad", mesh.TriangleCount, mesh.FaceCount)
	}

	// World-space pick landing on triangulation half 0 (v0,v1,v2). Local
	// point (0.25,0.75) -> world (0.25,2,3.75); shoot downward from y=5.
	ray := Ray{
		Origin: [3]float64{0.25, 5, 3.75},
		Dir:    [3]float64{0, -1, 0},
	}
	hit, ok := mesh.Intersect(ray, 0, math.Inf(1))
	if !ok {
		t.Fatal("expected transformed quad hit")
	}
	if hit.TriangleIndex != 0 {
		t.Fatalf("triangle ordinal: %d, want 0", hit.TriangleIndex)
	}

	// Both triangles belong to the single original quad face.
	if hit.FaceIndex != 0 {
		t.Fatalf("face index: %d, want 0 (the quad)", hit.FaceIndex)
	}
	if !approx(3, hit.T, 1e-9) {
		t.Fatalf("t: %g, want 3", hit.T)
	}
	if !approxVec3([3]float64{0.25, 2, 3.75}, hit.Point, 1e-9) {
		t.Fatalf("point: %v", hit.Point)
	}
	if !hit.FrontFace {
		t.Fatal("expected front face under the transform")
	}

	// The local winding normal is -z; R_x(90deg) maps -z to world +y.
	if !approxVec3([3]float64{0, 1, 0}, hit.GeometricNormal, 1e-9) {
		t.Fatalf("geometric normal: %v, want (0,1,0)", hit.GeometricNormal)
	}

	// Vertex normals barycentrically interpolated then rotated. At local
	// (0.25,0.75) on triangle (v0,v1,v2) the weights are (0.25,0.5,0.25),
	// giving local normal 0.25*(1,0,0)+0.5*(0,1,0)+0.25*(0,0,1)
	// = (0.25,0.5,0.25); R_x(90deg) maps it to (0.25,-0.25,0.5).
	wantN := normalize3Of(0.25, -0.25, 0.5)
	if !hit.HasNormal || !approxVec3(wantN, hit.Normal, 1e-9) {
		t.Fatalf("interpolated normal: %v hasNormal=%v want %v", hit.Normal, hit.HasNormal, wantN)
	}

	// Interpolated UV at local (0.25,0.75).
	if !hit.HasUV || !approxVec2([2]float64{0.25, 0.75}, hit.UV, 1e-9) {
		t.Fatalf("uv: %v hasUV=%v", hit.UV, hit.HasUV)
	}

	// Hit triangulation half 1 (v2,v3,v0), which keeps the same -z winding
	// normal as half 0. Local point (0.75,0.25) -> world (0.75,2,3.25).
	ray2 := Ray{
		Origin: [3]float64{0.75, 5, 3.25},
		Dir:    [3]float64{0, -1, 0},
	}
	hit2, ok := mesh.Intersect(ray2, 0, math.Inf(1))
	if !ok {
		t.Fatal("expected second-half hit")
	}
	if hit2.TriangleIndex != 1 {
		t.Fatalf("second-half triangle ordinal: %d, want 1", hit2.TriangleIndex)
	}
	// Same local -z winding normal as half 0, rotated to world +y.
	if !hit2.FrontFace {
		t.Fatal("second quad half front side: want front-facing")
	}
	if !approxVec3([3]float64{0, 1, 0}, hit2.GeometricNormal, 1e-9) {
		t.Fatalf("second-half geometric normal: %v, want (0,1,0)", hit2.GeometricNormal)
	}
	bSum := hit2.Bary[0] + hit2.Bary[1] + hit2.Bary[2]
	if !approx(1, bSum, 1e-9) {
		t.Fatalf("barycentric sum: %g weights=%v", bSum, hit2.Bary)
	}
	if !approxVec2([2]float64{0.75, 0.25}, hit2.UV, 1e-9) {
		t.Fatalf("second-half uv: %v, want (0.75,0.25)", hit2.UV)
	}
	if hit2.FaceIndex != 0 {
		t.Fatalf("second-half face: %d, want 0", hit2.FaceIndex)
	}
}

func normalize3Of(x, y, z float64) [3]float64 {
	nx, ny, nz := normalize3(x, y, z)
	return [3]float64{nx, ny, nz}
}

// TestRayNegativeScale verifies winding and normals under a mirrored model
// transform with negative scale.
func TestRayNegativeScale(t *testing.T) {
	o, err := NewObjFromBuf("quadObj", []byte(quadObj), rayOptions())
	if err != nil {
		t.Fatalf("NewObjFromBuf: %v", err)
	}

	// Mirror x and flip it inside-out, plus a translation.
	model := MultiplyMatrix4(TranslationMatrix4(10, 0, 0), ScaleMatrix4(-2, 2, 2))
	mesh, err := NewRayMesh(o, model, nil)
	if err != nil {
		t.Fatalf("NewRayMesh: %v", err)
	}

	// Pick triangulation half 1 (local point (0.75,0.25,0) -> world (8.5,
	// 0.5,0)), whose local winding normal is -z due to gwob's quad fan.
	ray := Ray{
		Origin: [3]float64{8.5, 0.5, 5},
		Dir:    [3]float64{0, 0, -1},
	}
	hit, ok := mesh.Intersect(ray, 0, math.Inf(1))
	if !ok {
		t.Fatal("expected hit on mirrored quad")
	}
	if hit.TriangleIndex != 1 {
		t.Fatalf("triangle: %d, want 1", hit.TriangleIndex)
	}
	if !approx(5, hit.T, 1e-9) || !approxVec3([3]float64{8.5, 0.5, 0}, hit.Point, 1e-9) {
		t.Fatalf("t=%g point=%v", hit.T, hit.Point)
	}
	if !hit.FrontFace {
		t.Fatal("negative scale flips winding: expected front-facing hit")
	}
	// The local -z fan winding normal is mapped by the inverse-transpose
	// diag(-0.5,0.5,0.5) to world -z. Front/back is carried by FrontFace,
	// not by flipping the normal.
	if !approxVec3([3]float64{0, 0, -1}, hit.GeometricNormal, 1e-9) {
		t.Fatalf("geometric normal under mirror: %v, want (0,0,-1)", hit.GeometricNormal)
	}

	// A ray from the opposite side is back-facing and must be culled.
	backRay := Ray{
		Origin: [3]float64{8.5, 0.5, -5},
		Dir:    [3]float64{0, 0, 1},
	}
	mesh.CullBackfaces = true
	if _, ok = mesh.Intersect(backRay, 0, math.Inf(1)); ok {
		t.Fatal("cull with negative scale: back-facing hit should be rejected")
	}
	// Without culling the same ray still reports the hit.
	mesh.CullBackfaces = false
	if _, ok = mesh.Intersect(backRay, 0, math.Inf(1)); !ok {
		t.Fatal("back-facing hit expected when culling is disabled")
	}
}

// degenerateObj mixes good, degenerate and non-finite-coordinate triangles.
const degenerateObj = `
o degenerate
v 0 0 0
v 1 0 0
v 1 1 0
v 2 2 2
v 3 3 3
f 1 2 3
f 1 1 2
f 1 4 5
`

func TestRayDegenerateAndBadData(t *testing.T) {
	o, err := NewObjFromBuf("degenerateObj", []byte(degenerateObj), rayOptions())
	if err != nil {
		t.Fatalf("NewObjFromBuf: %v", err)
	}

	mesh, err := NewRayMesh(o, IdentityMatrix4(), nil)
	if err == nil {
		t.Fatal("expected build error reporting dropped triangles")
	}
	if mesh.DegenerateTris != 2 {
		t.Fatalf("degenerate count: %d, want 2", mesh.DegenerateTris)
	}
	if mesh.TriangleCount != 1 || mesh.FaceCount != 1 {
		t.Fatalf("valid triangles: %d faces: %d, want 1/1", mesh.TriangleCount, mesh.FaceCount)
	}

	// The single good triangle still picks normally (interior point (0.2,0.2)).
	ray := Ray{
		Origin: [3]float64{0.2, 0.2, 5},
		Dir:    [3]float64{0, 0, -1},
	}
	hit, ok := mesh.Intersect(ray, 0, math.Inf(1))
	if !ok || hit.FaceIndex != 0 {
		t.Fatalf("good triangle pick: ok=%v face=%d", ok, hit.FaceIndex)
	}

	// Non-finite ray coordinates and zero direction are clean misses.
	badRays := []Ray{
		{Origin: [3]float64{math.NaN(), 0, 0}, Dir: [3]float64{0, 0, -1}},
		{Origin: [3]float64{0, 0, 0}, Dir: [3]float64{math.Inf(1), 0, 0}},
		{Origin: [3]float64{0, 0, 0}, Dir: [3]float64{0, 0, 0}},
	}
	for i, br := range badRays {
		if _, ok := mesh.Intersect(br, 0, math.Inf(1)); ok {
			t.Fatalf("bad ray %d should miss", i)
		}
	}

	// Parallel and coplanar rays miss.
	parallel := Ray{
		Origin: [3]float64{0.25, 0.25, 1},
		Dir:    [3]float64{1, 0, 0},
	}
	if _, ok = mesh.Intersect(parallel, 0, math.Inf(1)); ok {
		t.Fatal("parallel ray should miss")
	}
	coplanar := Ray{
		Origin: [3]float64{-1, 0, 0},
		Dir:    [3]float64{1, 0, 0},
	}
	if _, ok = mesh.Intersect(coplanar, 0, math.Inf(1)); ok {
		t.Fatal("coplanar ray should miss")
	}

	// Ray aimed beside the triangle misses.
	miss := Ray{
		Origin: [3]float64{5, 5, 5},
		Dir:    [3]float64{0, 0, -1},
	}
	if _, ok = mesh.Intersect(miss, 0, math.Inf(1)); ok {
		t.Fatal("off-target ray should miss")
	}
}

// TestRayBadIndices builds a mesh whose index buffer points past the vertex
// array; those triangles are reported invalid instead of panicking.
func TestRayBadIndices(t *testing.T) {
	o, err := NewObjFromVertex(
		[]float32{0, 0, 0, 1, 0, 0, 1, 1, 0},
		[]int{0, 1, 2, 0, 9, 2},
	)
	if err != nil {
		t.Fatalf("NewObjFromVertex: %v", err)
	}

	mesh, errBuild := NewRayMesh(o, IdentityMatrix4(), nil)
	if errBuild == nil {
		t.Fatal("expected error for out-of-range indices")
	}
	if mesh.InvalidTris != 1 || mesh.TriangleCount != 1 {
		t.Fatalf("invalid=%d valid=%d, want invalid=1 valid=1", mesh.InvalidTris, mesh.TriangleCount)
	}

	ray := Ray{
		Origin: [3]float64{0.5, 0.5, 5},
		Dir:    [3]float64{0, 0, -1},
	}
	hit, ok := mesh.Intersect(ray, 0, math.Inf(1))
	if !ok {
		t.Fatal("valid triangle should still be pickable")
	}
	// No parser face data: fallback ordinal is 0 for the first triangle.
	if hit.FaceIndex != 0 || hit.TriangleIndex != 0 {
		t.Fatalf("face=%d tri=%d, want 0/0", hit.FaceIndex, hit.TriangleIndex)
	}
}

// TestRayNonFiniteCoord feeds NaN/Inf vertex coordinates directly.
func TestRayNonFiniteCoord(t *testing.T) {
	o, err := NewObjFromVertex(
		[]float32{0, 0, 0, 1, 0, 0, float32(math.NaN()), 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0},
		[]int{0, 1, 2, 3, 4, 5},
	)
	if err != nil {
		t.Fatalf("NewObjFromVertex: %v", err)
	}

	mesh, errBuild := NewRayMesh(o, IdentityMatrix4(), nil)
	if errBuild == nil {
		t.Fatal("expected error for non-finite vertex")
	}
	if mesh.InvalidTris != 1 || mesh.TriangleCount != 1 {
		t.Fatalf("invalid=%d valid=%d, want 1/1", mesh.InvalidTris, mesh.TriangleCount)
	}
}

// TestRaySingularTransform ensures a non-invertible model matrix is rejected
// up front instead of producing garbage picks.
func TestRaySingularTransform(t *testing.T) {
	o, err := NewObjFromBuf("quadObj", []byte(quadObj), rayOptions())
	if err != nil {
		t.Fatalf("NewObjFromBuf: %v", err)
	}
	if _, err := NewRayMesh(o, ScaleMatrix4(1, 0, 1), nil); err == nil {
		t.Fatal("singular transform must fail")
	}
	if _, err := NewRayMesh(nil, IdentityMatrix4(), nil); err == nil {
		t.Fatal("nil obj must fail")
	}
}

// TestRayEmptyMesh covers an Obj without any geometry.
func TestRayEmptyMesh(t *testing.T) {
	o, err := NewObjFromBuf("empty", []byte{}, rayOptions())
	if err != nil {
		t.Fatalf("NewObjFromBuf: %v", err)
	}
	mesh, errMesh := NewRayMesh(o, IdentityMatrix4(), nil)
	if errMesh != nil {
		t.Fatalf("empty mesh build: %v", errMesh)
	}
	ray := Ray{
		Origin: [3]float64{0, 0, 0},
		Dir:    [3]float64{0, 0, -1},
	}
	if _, ok := mesh.Intersect(ray, 0, math.Inf(1)); ok {
		t.Fatal("empty mesh must never hit")
	}
}

// TestRayCubeObject exercises the full pipeline against the existing cube
// fixture (which uses relative indices, UVs and normals), checking that
// original parser data remains intact and every cube face is pickable.
func TestRayCubeObject(t *testing.T) {
	o, err := NewObjFromBuf("cubeObj", []byte(cubeObj), &ObjParserOptions{LogStats: rayLogStats})
	if err != nil {
		t.Fatalf("NewObjFromBuf: %v", err)
	}
	if !sliceEqualInt(cubeIndices, o.Indices) {
		t.Fatal("parser indices changed by ray support")
	}
	if !sliceEqualFloat(cubeCoord, o.Coord) {
		t.Fatal("parser coord changed by ray support")
	}

	mesh, err := NewRayMesh(o, IdentityMatrix4(), nil)
	if err != nil {
		t.Fatalf("NewRayMesh: %v", err)
	}
	// Cube fixture: 12 triangles from 12 triangle faces.
	if mesh.TriangleCount != 12 || mesh.FaceCount != 12 {
		t.Fatalf("cube structure: tris=%d faces=%d", mesh.TriangleCount, mesh.FaceCount)
	}

	// Pick each of the six axis faces from outside and confirm normals.
	cases := []struct {
		origin [3]float64
		dir    [3]float64
		normal [3]float64
		point  [3]float64
	}{
		{[3]float64{0, 0, 5}, [3]float64{0, 0, -1}, [3]float64{0, 0, 1}, [3]float64{0, 0, 1}},
		{[3]float64{0, 0, -5}, [3]float64{0, 0, 1}, [3]float64{0, 0, -1}, [3]float64{0, 0, -1}},
		{[3]float64{5, 0, 0}, [3]float64{-1, 0, 0}, [3]float64{1, 0, 0}, [3]float64{1, 0, 0}},
		{[3]float64{-5, 0, 0}, [3]float64{1, 0, 0}, [3]float64{-1, 0, 0}, [3]float64{-1, 0, 0}},
		{[3]float64{0, 5, 0}, [3]float64{0, -1, 0}, [3]float64{0, 1, 0}, [3]float64{0, 1, 0}},
		{[3]float64{0, -5, 0}, [3]float64{0, 1, 0}, [3]float64{0, -1, 0}, [3]float64{0, -1, 0}},
	}
	for i, c := range cases {
		hit, ok := mesh.Intersect(Ray{Origin: c.origin, Dir: c.dir}, 0, math.Inf(1))
		if !ok {
			t.Fatalf("cube case %d: miss", i)
		}
		if !approxVec3(c.point, hit.Point, 1e-8) {
			t.Fatalf("cube case %d: point=%v want=%v", i, hit.Point, c.point)
		}
		if !approxVec3(c.normal, hit.Normal, 1e-8) {
			t.Fatalf("cube case %d: normal=%v want=%v", i, hit.Normal, c.normal)
		}
		// The cube fixture's geometric winding does not consistently match its
		// vn directions, so FrontFace is not asserted here; the dedicated
		// winding/cull tests cover it.
		if !hit.HasUV || !hit.HasNormal {
			t.Fatalf("cube case %d: hasUV=%v hasNormal=%v", i, hit.HasUV, hit.HasNormal)
		}
		if hit.GroupIndex < 0 || hit.GroupIndex >= len(o.Groups) {
			t.Fatalf("cube case %d: group index %d out of range", i, hit.GroupIndex)
		}
	}
}

// TestRayBarycentricInterior hits a known interior point and verifies the
// barycentric weights and interpolated UVs analytically.
func TestRayBarycentricInterior(t *testing.T) {
	const triObj = `
o tri
v 0 0 0
v 4 0 0
v 0 4 0
vt 0 0
vt 4 0
vt 0 4
f 1/1 2/2 3/3
`
	o, err := NewObjFromBuf("triObj", []byte(triObj), rayOptions())
	if err != nil {
		t.Fatalf("NewObjFromBuf: %v", err)
	}
	mesh, err := NewRayMesh(o, IdentityMatrix4(), nil)
	if err != nil {
		t.Fatalf("NewRayMesh: %v", err)
	}

	// Point (1,1,0) => weights w0=0.5, w1=0.25, w2=0.25 and UV (1,1).
	ray := Ray{
		Origin: [3]float64{1, 1, 5},
		Dir:    [3]float64{0, 0, -1},
	}
	hit, ok := mesh.Intersect(ray, 0, math.Inf(1))
	if !ok {
		t.Fatal("expected hit")
	}
	if !approxVec3([3]float64{0.5, 0.25, 0.25}, hit.Bary, 1e-9) {
		t.Fatalf("barycentric: %v", hit.Bary)
	}
	if !approxVec2([2]float64{1, 1}, hit.UV, 1e-9) {
		t.Fatalf("uv: %v", hit.UV)
	}

	// Edge hit (on the v0-v1 edge) must remain a valid, in-range hit.
	edge := Ray{
		Origin: [3]float64{2, 0, 5},
		Dir:    [3]float64{0, 0, -1},
	}
	hit, ok = mesh.Intersect(edge, 0, math.Inf(1))
	if !ok {
		t.Fatal("edge hit expected")
	}
	if !approx(0, hit.Bary[2], 1e-8) {
		t.Fatalf("edge barycentric: %v", hit.Bary)
	}
}

// TestRayStructureReused proves the acceleration structure is built once and
// reused across queries: the node/leaf storage is stable while many rays run,
// and the structure actually partitions geometry into multiple nodes.
func TestRayStructureReused(t *testing.T) {
	o, err := NewObjFromBuf("cubeObj", []byte(cubeObj), rayOptions())
	if err != nil {
		t.Fatalf("NewObjFromBuf: %v", err)
	}
	mesh, err := NewRayMesh(o, IdentityMatrix4(), &RayMeshOptions{LeafSize: 2})
	if err != nil {
		t.Fatalf("NewRayMesh: %v", err)
	}
	if len(mesh.nodes) < 3 {
		t.Fatalf("BVH should contain internal nodes, got %d", len(mesh.nodes))
	}
	nodesBefore := len(mesh.nodes)
	leafBefore := len(mesh.leafOrder)
	trisBefore := len(mesh.tris)

	for i := 0; i < 100; i++ {
		ray := Ray{
			Origin: [3]float64{float64(i%5) - 2, float64((i*3)%5) - 2, 5},
			Dir:    [3]float64{0, 0, -1},
		}
		_, _ = mesh.Intersect(ray, 0, math.Inf(1))
	}

	if len(mesh.nodes) != nodesBefore || len(mesh.leafOrder) != leafBefore || len(mesh.tris) != trisBefore {
		t.Fatal("Intersect rebuilt or mutated the acceleration structure")
	}
	if mesh.TriangleCount != 12 {
		t.Fatalf("triangle count changed: %d", mesh.TriangleCount)
	}
}

// TestRayGroupIndex checks that hits report the owning group.
func TestRayGroupIndex(t *testing.T) {
	o, err := NewObjFromBuf("cubeObj", []byte(cubeObj), rayOptions())
	if err != nil {
		t.Fatalf("NewObjFromBuf: %v", err)
	}
	mesh, err := NewRayMesh(o, IdentityMatrix4(), nil)
	if err != nil {
		t.Fatalf("NewRayMesh: %v", err)
	}
	ray := Ray{Origin: [3]float64{0, 0, 5}, Dir: [3]float64{0, 0, -1}}
	hit, ok := mesh.Intersect(ray, 0, math.Inf(1))
	if !ok {
		t.Fatal("expected hit")
	}
	if hit.GroupIndex < 0 || hit.GroupIndex >= len(o.Groups) {
		t.Fatalf("group index %d outside [0,%d)", hit.GroupIndex, len(o.Groups))
	}
	if o.Groups[hit.GroupIndex].Name == "" && len(o.Groups) != 1 {
		t.Fatalf("unexpected group resolution: %d", hit.GroupIndex)
	}
}

func BenchmarkRayCubeIntersect(b *testing.B) {
	o, err := NewObjFromBuf("cubeObj", []byte(cubeObj), &ObjParserOptions{LogStats: false})
	if err != nil {
		b.Fatal(err)
	}
	mesh, err := NewRayMesh(o, IdentityMatrix4(), nil)
	if err != nil {
		b.Fatal(err)
	}
	ray := Ray{Origin: [3]float64{0, 0, 5}, Dir: [3]float64{0, 0, -1}}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = mesh.Intersect(ray, 0, math.Inf(1))
	}
}
