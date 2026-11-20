/*
Mesh ray picking for Wavefront .OBJ geometry.

A RayMesh is a reusable acceleration structure (bounding volume hierarchy)
built once from a parsed Obj and queried many times with pick rays. Repeated
Intersect calls never rebuild the index: call NewRayMesh once per model and
keep the returned *RayMesh.

Example:

	mesh, err := gwob.NewRayMesh(obj, gwob.IdentityMatrix4(), nil)
	if err != nil {
		// some triangles were unusable, see mesh.InvalidTris
	}

	ray := gwob.Ray{
		Origin: [3]float64{0, 0, 5},
		Dir:    [3]float64{0, 0, -1},
	}

	hit, ok := mesh.Intersect(ray, 0, math.Inf(1))
	if ok {
		// hit.FaceIndex, hit.Bary, hit.UV, hit.Normal ...
	}
*/
package gwob

import (
	"fmt"
	"math"
	"sort"
)

// Ray is a half-line Origin + t*Dir, t >= 0. Dir does not need to be
// normalized: hit T is measured in units of Dir.
type Ray struct {
	Origin [3]float64
	Dir    [3]float64
}

// RayHit describes the nearest intersection of a ray with one triangle.
type RayHit struct {
	T float64 // ray parameter of the hit: Point = Origin + T*Dir

	Point [3]float64 // hit point in world (model) space

	// Bary holds the barycentric weights (w0, w1, w2) of the three triangle
	// vertices referenced by TriangleIndex, so w0+w1+w2 = 1.
	Bary [3]float64

	// Normal is the interpolated shading normal in world space. When the OBJ
	// provides vertex normals (vn), they are barycentrically interpolated and
	// transformed by the inverse-transpose model matrix; otherwise the
	// geometric triangle normal is used.
	Normal [3]float64

	// GeometricNormal is the triangle winding normal in world space, always
	// available even when vertex normals exist. It is the local winding
	// normal transformed by the inverse-transpose model matrix; under a
	// mirrored (negative-determinant) transform its orientation is preserved
	// while FrontFace reports the reversed front/back classification.
	GeometricNormal [3]float64

	// UV is the barycentric-interpolated texture coordinate. HasUV reports
	// whether the OBJ provides texture coordinates.
	UV    [2]float64
	HasUV bool

	// HasNormal reports whether Normal comes from interpolated vertex normals.
	HasNormal bool

	// FaceIndex is the 0-based ordinal of the original OBJ face (f line),
	// counting only faces accepted by the parser. A quad face spans two
	// triangles that share the same FaceIndex.
	FaceIndex int

	// TriangleIndex is the 0-based ordinal of the triangle inside Obj.Indices
	// (three indices per triangle; a quad produces two consecutive triangles).
	TriangleIndex int

	// GroupIndex is the 0-based ordinal of the Obj.Groups entry the triangle
	// belongs to, or -1 when it belongs to no group.
	GroupIndex int

	// FrontFace reports whether the ray hit the front side of the triangle,
	// taking the model transform handedness (negative scale) into account.
	FrontFace bool
}

// RayMeshOptions customizes acceleration structure construction.
type RayMeshOptions struct {
	// CullBackfaces makes Intersect ignore back-facing triangles. The value
	// is copied onto RayMesh.CullBackfaces, which may be toggled later without
	// rebuilding the structure.
	CullBackfaces bool

	// LeafSize bounds the number of triangles per BVH leaf. Zero selects a
	// sensible default.
	LeafSize int
}

// rayTri is one indexable, validated triangle of the mesh.
type rayTri struct {
	i0, i1, i2 int // stride element indices into Obj.Coord

	faceIndex int // original OBJ face ordinal
	triIndex  int // triangle ordinal within Obj.Indices
	group     int // Obj.Groups ordinal, -1 if none
}

// bvhNode is a BVH node. Count > 0 marks a leaf node.
type bvhNode struct {
	min, max     [3]float64
	left, right  int32 // internal node children
	start, count int32 // leaf node range into RayMesh.leafOrder
}

// RayMesh is a reusable BVH acceleration structure built from an Obj. The
// original parsed Obj data and its public fields are left untouched.
type RayMesh struct {
	obj   *Obj
	model Matrix4    // model (local->world) transform
	inv   Matrix4    // inverse model transform
	nMat  [9]float64 // inverse-transpose upper 3x3, row-major
	det   float64    // determinant of the model 3x3

	tris      []rayTri  // valid triangles
	leafOrder []int32   // triangle ids, concatenated in leaf order
	nodes     []bvhNode // node pool, root is nodes[0]

	// CullBackfaces skips back-facing triangles during Intersect when true.
	// It may be changed at any time without rebuilding the structure.
	CullBackfaces bool

	// LeafSize is the maximum number of triangles stored in a BVH leaf.
	LeafSize int

	// TriangleCount is the number of valid, intersectable triangles.
	TriangleCount int

	// FaceCount is the number of distinct original OBJ faces represented.
	FaceCount int

	// InvalidTris counts triangles dropped because of out-of-range indices or
	// non-finite vertex coordinates.
	InvalidTris int

	// DegenerateTris counts zero-area triangles dropped at build time. They
	// can never produce a finite hit.
	DegenerateTris int
}

const (
	defaultLeafSize = 4
	baryTol         = 1e-10
	parallelTol     = 1e-12
	maxBVHDepth     = 64
)

// NewRayMesh builds a reusable ray picking acceleration structure for obj.
// model is the local-to-world transform applied to the mesh (use
// IdentityMatrix4 for the raw OBJ coordinates); it must be invertible.
//
// The structure is built exactly once: pass the same *RayMesh to every
// Intersect call. A non-nil error is returned when the transform is singular
// or when some triangles were unusable (bad indices, non-finite or
// degenerate data); the returned mesh still contains every valid triangle
// and remains usable.
func NewRayMesh(obj *Obj, model Matrix4, options *RayMeshOptions) (*RayMesh, error) {
	if obj == nil {
		return nil, fmt.Errorf("NewRayMesh: nil obj")
	}

	inv, err := InverseMatrix4(model)
	if err != nil {
		return nil, fmt.Errorf("NewRayMesh: singular model transform: %w", err)
	}

	m := &RayMesh{
		obj:   obj,
		model: model,
		inv:   inv,
		det:   det3(model),
	}
	m.nMat = normalMatrix3(inv)

	if options != nil {
		m.CullBackfaces = options.CullBackfaces
		m.LeafSize = options.LeafSize
	}
	if m.LeafSize <= 0 {
		m.LeafSize = defaultLeafSize
	}

	if errBuild := m.build(); errBuild != nil {
		return m, errBuild
	}

	return m, nil
}

// stridePos returns the local-space position of stride element i.
func (m *RayMesh) stridePos(i int) (float64, float64, float64) {
	off := m.obj.StrideOffsetPosition / 4
	stride := m.obj.StrideSize / 4
	f := off + i*stride
	return float64(m.obj.Coord[f]), float64(m.obj.Coord[f+1]), float64(m.obj.Coord[f+2])
}

// strideUV returns the texture coordinate of stride element i.
func (m *RayMesh) strideUV(i int) (float64, float64) {
	off := m.obj.StrideOffsetTexture / 4
	stride := m.obj.StrideSize / 4
	f := off + i*stride
	return float64(m.obj.Coord[f]), float64(m.obj.Coord[f+1])
}

// strideNormal returns the vertex normal of stride element i.
func (m *RayMesh) strideNormal(i int) (float64, float64, float64) {
	off := m.obj.StrideOffsetNormal / 4
	stride := m.obj.StrideSize / 4
	f := off + i*stride
	return float64(m.obj.Coord[f]), float64(m.obj.Coord[f+1]), float64(m.obj.Coord[f+2])
}

// groupForTri maps a triangle ordinal to its Obj.Groups ordinal.
func groupForTri(o *Obj, tri int) int {
	g := -1
	for i, gr := range o.Groups {
		begin := gr.IndexBegin / 3
		end := begin + gr.IndexCount/3
		if tri >= begin && tri < end {
			g = i
		}
	}
	return g
}

// faceOrdinals maps each triangle ordinal to the original OBJ face ordinal.
func faceOrdinals(o *Obj, numTris int) []int {
	faceOfTri := make([]int, numTris)
	if numTris == 0 {
		return faceOfTri
	}

	if len(o.faceSizes) > 0 {
		// Trust the face sizes recorded by the parser.
		k := 0
		for fi, size := range o.faceSizes {
			n := size - 2 // triangles produced by a triangle (1) or quad (2)
			if n < 1 {
				n = 1
			}
			for j := 0; j < n && k < numTris; j++ {
				faceOfTri[k] = fi
				k++
			}
		}
		for ; k < numTris; k++ {
			faceOfTri[k] = len(o.faceSizes)
		}
		return faceOfTri
	}

	// Fallback for meshes assembled through NewObjFromVertex: detect the
	// quad triangulation pattern (v0 v1 v2),(v2 v3 v0).
	faceOfTri[0] = 0
	ind := o.Indices
	face := 0
	for k := 1; k < numTris; k++ {
		b := 3 * k
		a := b - 3
		isQuadSecond := ind[b] == ind[a+2] && // v2
			ind[b+2] == ind[a] && // v0
			ind[b+1] != ind[a+1] // distinct v3
		if !isQuadSecond {
			face++
		}
		faceOfTri[k] = face
	}
	return faceOfTri
}

func finite3(a, b, c float64) bool {
	return !math.IsInf(a, 0) && !math.IsNaN(a) &&
		!math.IsInf(b, 0) && !math.IsNaN(b) &&
		!math.IsInf(c, 0) && !math.IsNaN(c)
}

func triangleArea(ax, ay, az, bx, by, bz, cx, cy, cz float64) float64 {
	e1x, e1y, e1z := bx-ax, by-ay, bz-az
	e2x, e2y, e2z := cx-ax, cy-ay, cz-az
	cx2 := e1y*e2z - e1z*e2y
	cy2 := e1z*e2x - e1x*e2z
	cz2 := e1x*e2y - e1y*e2x
	return 0.5 * math.Sqrt(cx2*cx2+cy2*cy2+cz2*cz2)
}

func (m *RayMesh) build() error {
	o := m.obj
	numTris := len(o.Indices) / 3
	faceOfTri := faceOrdinals(o, numTris)

	elementCount := 0
	if o.StrideSize > 0 {
		elementCount = o.NumberOfElements()
	}

	maxFace := -1
	for k := 0; k < numTris; k++ {
		i0, i1, i2 := o.Indices[3*k], o.Indices[3*k+1], o.Indices[3*k+2]

		if i0 < 0 || i1 < 0 || i2 < 0 || i0 >= elementCount || i1 >= elementCount || i2 >= elementCount {
			m.InvalidTris++
			continue
		}

		x0, y0, z0 := m.stridePos(i0)
		x1, y1, z1 := m.stridePos(i1)
		x2, y2, z2 := m.stridePos(i2)

		if !finite3(x0, y0, z0) || !finite3(x1, y1, z1) || !finite3(x2, y2, z2) {
			m.InvalidTris++
			continue
		}

		// Degenerate (zero-area) triangles can never produce a finite hit.
		if triangleArea(x0, y0, z0, x1, y1, z1, x2, y2, z2) == 0 {
			m.DegenerateTris++
			continue
		}

		fi := faceOfTri[k]
		if fi > maxFace {
			maxFace = fi
		}

		m.tris = append(m.tris, rayTri{
			i0:        i0,
			i1:        i1,
			i2:        i2,
			faceIndex: fi,
			triIndex:  k,
			group:     groupForTri(o, k),
		})
	}

	m.TriangleCount = len(m.tris)
	m.FaceCount = maxFace + 1

	leftover := len(o.Indices) - numTris*3

	if len(m.tris) == 0 {
		m.nodes = []bvhNode{{}}
	} else {
		ids := make([]int32, len(m.tris))
		for i := range ids {
			ids[i] = int32(i)
		}
		m.nodes = make([]bvhNode, 0, 2*len(m.tris))
		m.buildRec(ids, 0)
	}

	if m.InvalidTris > 0 || m.DegenerateTris > 0 || leftover > 0 {
		return fmt.Errorf("NewRayMesh: dropped geometry: invalid_tris=%d degenerate_tris=%d dangling_indices=%d",
			m.InvalidTris, m.DegenerateTris, leftover)
	}
	return nil
}

func (m *RayMesh) centroid(t rayTri) [3]float64 {
	ax, ay, az := m.stridePos(t.i0)
	bx, by, bz := m.stridePos(t.i1)
	cx, cy, cz := m.stridePos(t.i2)
	return [3]float64{(ax + bx + cx) / 3, (ay + by + cy) / 3, (az + bz + cz) / 3}
}

// buildRec recursively constructs the BVH and returns the node index.
func (m *RayMesh) buildRec(ids []int32, depth int) int32 {
	node := bvhNode{
		min: [3]float64{math.Inf(1), math.Inf(1), math.Inf(1)},
		max: [3]float64{math.Inf(-1), math.Inf(-1), math.Inf(-1)},
	}

	for _, id := range ids {
		t := m.tris[id]
		for _, vi := range [3]int{t.i0, t.i1, t.i2} {
			x, y, z := m.stridePos(vi)
			if x < node.min[0] {
				node.min[0] = x
			}
			if y < node.min[1] {
				node.min[1] = y
			}
			if z < node.min[2] {
				node.min[2] = z
			}
			if x > node.max[0] {
				node.max[0] = x
			}
			if y > node.max[1] {
				node.max[1] = y
			}
			if z > node.max[2] {
				node.max[2] = z
			}
		}
	}

	if len(ids) <= m.LeafSize || depth >= maxBVHDepth {
		node.start = int32(len(m.leafOrder))
		node.count = int32(len(ids))
		m.leafOrder = append(m.leafOrder, ids...)
		m.nodes = append(m.nodes, node)
		return int32(len(m.nodes) - 1)
	}

	// Split along the largest centroid extent, at the median.
	var cMin, cMax [3]float64
	cMin = [3]float64{math.Inf(1), math.Inf(1), math.Inf(1)}
	for _, id := range ids {
		c := m.centroid(m.tris[id])
		for a := 0; a < 3; a++ {
			if c[a] < cMin[a] {
				cMin[a] = c[a]
			}
			if c[a] > cMax[a] {
				cMax[a] = c[a]
			}
		}
	}
	axis := 0
	if cMax[1]-cMin[1] > cMax[axis]-cMin[axis] {
		axis = 1
	}
	if cMax[2]-cMin[2] > cMax[axis]-cMin[axis] {
		axis = 2
	}

	sort.Slice(ids, func(i, j int) bool {
		ci := m.centroid(m.tris[ids[i]])
		cj := m.centroid(m.tris[ids[j]])
		return ci[axis] < cj[axis]
	})
	mid := len(ids) / 2

	m.nodes = append(m.nodes, node)
	idx := int32(len(m.nodes) - 1)
	left := m.buildRec(ids[:mid], depth+1)
	right := m.buildRec(ids[mid:], depth+1)
	m.nodes[idx].left = left
	m.nodes[idx].right = right
	return idx
}

// Intersect finds the nearest intersection of the ray with the mesh in the
// t interval [tmin, tmax]. The boolean result is false when nothing is hit.
// The acceleration structure is never rebuilt.
func (m *RayMesh) Intersect(ray Ray, tmin, tmax float64) (RayHit, bool) {
	var miss RayHit

	if m.TriangleCount == 0 {
		return miss, false
	}
	if !finite3(ray.Origin[0], ray.Origin[1], ray.Origin[2]) ||
		!finite3(ray.Dir[0], ray.Dir[1], ray.Dir[2]) ||
		ray.Dir == [3]float64{} {
		return miss, false // non-finite or zero-length ray
	}
	if math.IsNaN(tmin) || math.IsNaN(tmax) || tmin < 0 || tmax < tmin {
		return miss, false
	}

	// Bring the ray into mesh-local space. Because invModel is affine,
	// invModel(O + t*D) = invModel*O + t*invModel*D, so the local and world
	// ray parameter t are identical.
	lo := transformPoint(m.inv, ray.Origin)
	ld := transformVector(m.inv, ray.Dir)

	var best rayTri
	bestFront := false
	found := false
	bestT := tmax

	var stack [maxBVHDepth + 2]int32
	sp := 0
	stack[0] = 0
	sp++

	for sp > 0 {
		sp--
		n := &m.nodes[stack[sp]]

		if n.count > 0 {
			end := n.start + n.count
			for li := n.start; li < end; li++ {
				tri := m.tris[m.leafOrder[li]]
				if hit, front, ok := m.intersectTri(tri, lo, ld, tmin, bestT); ok {
					best = tri
					bestT = hit
					bestFront = front
					found = true
				}
			}
			continue
		}

		l := &m.nodes[n.left]
		r := &m.nodes[n.right]
		if hitBBox(l, lo, ld, tmin, bestT) {
			stack[sp] = n.left
			sp++
		}
		if hitBBox(r, lo, ld, tmin, bestT) {
			stack[sp] = n.right
			sp++
		}
	}

	if !found {
		return miss, false
	}

	return m.fillHit(best, ray, lo, ld, bestT, bestFront), true
}

// hitBBox performs a slab test of the ray against an axis-aligned box.
func hitBBox(n *bvhNode, o, d [3]float64, tmin, tmax float64) bool {
	t0, t1 := tmin, tmax
	for a := 0; a < 3; a++ {
		if d[a] == 0 {
			if o[a] < n.min[a] || o[a] > n.max[a] {
				return false
			}
			continue
		}
		inv := 1 / d[a]
		near := (n.min[a] - o[a]) * inv
		far := (n.max[a] - o[a]) * inv
		if near > far {
			near, far = far, near
		}
		if near > t0 {
			t0 = near
		}
		if far < t1 {
			t1 = far
		}
		if t0 > t1 {
			return false
		}
	}
	return true
}

// intersectTri runs the Möller–Trumbore test in mesh-local space and returns
// the local (== world) ray parameter t and whether the hit is front-facing.
func (m *RayMesh) intersectTri(tri rayTri, o, d [3]float64, tmin, tmax float64) (float64, bool, bool) {
	ax, ay, az := m.stridePos(tri.i0)
	bx, by, bz := m.stridePos(tri.i1)
	cx, cy, cz := m.stridePos(tri.i2)

	e1x, e1y, e1z := bx-ax, by-ay, bz-az
	e2x, e2y, e2z := cx-ax, cy-ay, cz-az

	hx := d[1]*e2z - d[2]*e2y
	hy := d[2]*e2x - d[0]*e2z
	hz := d[0]*e2y - d[1]*e2x

	det := e1x*hx + e1y*hy + e1z*hz

	// Scale-aware parallel-ray test: det has units of length^2 here.
	eps := parallelTol * (math.Hypot(math.Hypot(e1x, e1y), e1z) +
		math.Hypot(math.Hypot(e2x, e2y), e2z)) *
		math.Hypot(math.Hypot(d[0], d[1]), d[2])
	if math.Abs(det) <= eps {
		return 0, false, false // ray parallel or coplanar with the triangle
	}

	// With h = d x e2, det = e1.(d x e2) = -d.(e1 x e2) = -d.n where n is
	// the winding normal. A front hit means the ray travels against n
	// (d.n < 0), i.e. det > 0. A negative model determinant (mirroring /
	// negative scale) reverses the winding, so multiply by m.det.
	front := det*m.det > 0
	if m.CullBackfaces && !front {
		return 0, false, false
	}

	f := 1 / det
	sx, sy, sz := o[0]-ax, o[1]-ay, o[2]-az
	u := f * (sx*hx + sy*hy + sz*hz)
	if u < -baryTol || u > 1+baryTol {
		return 0, false, false
	}

	qx := sy*e1z - sz*e1y
	qy := sz*e1x - sx*e1z
	qz := sx*e1y - sy*e1x
	v := f * (d[0]*qx + d[1]*qy + d[2]*qz)
	if v < -baryTol || u+v > 1+baryTol {
		return 0, false, false
	}

	t := f * (e2x*qx + e2y*qy + e2z*qz)
	if t < tmin || t > tmax {
		return 0, false, false
	}
	return t, front, true
}

// fillHit computes all interpolated, world-space hit attributes.
func (m *RayMesh) fillHit(tri rayTri, world Ray, lo, ld [3]float64, t float64, front bool) RayHit {
	ax, ay, az := m.stridePos(tri.i0)
	bx, by, bz := m.stridePos(tri.i1)
	cx, cy, cz := m.stridePos(tri.i2)

	e1x, e1y, e1z := bx-ax, by-ay, bz-az
	e2x, e2y, e2z := cx-ax, cy-ay, cz-az

	// Recover barycentric weights from the local hit point.
	px, py, pz := lo[0]+t*ld[0], lo[1]+t*ld[1], lo[2]+t*ld[2]
	u, v := barycentric(px, py, pz, ax, ay, az,
		e1x, e1y, e1z, e2x, e2y, e2z)
	w0 := 1 - u - v
	if w0 < 0 && w0 > -baryTol {
		w0 = 0
	}

	hit := RayHit{
		T:             t,
		Point:         [3]float64{world.Origin[0] + t*world.Dir[0], world.Origin[1] + t*world.Dir[1], world.Origin[2] + t*world.Dir[2]},
		Bary:          [3]float64{w0, u, v},
		FaceIndex:     tri.faceIndex,
		TriangleIndex: tri.triIndex,
		GroupIndex:    tri.group,
		FrontFace:     front,
	}

	// Geometric normal: winding normal in local space, transformed to world
	// space by the inverse-transpose model matrix. A mirrored transform
	// (det < 0) flips the shading normal automatically through that matrix;
	// FrontFace carries the winding information instead.
	nlx, nly, nlz := crossNormalize(e1x, e1y, e1z, e2x, e2y, e2z)
	gx, gy, gz := mulMat3(m.nMat, nlx, nly, nlz)
	gx, gy, gz = normalize3(gx, gy, gz)
	hit.GeometricNormal = [3]float64{gx, gy, gz}

	if m.obj.NormCoordFound {
		n0x, n0y, n0z := m.strideNormal(tri.i0)
		n1x, n1y, n1z := m.strideNormal(tri.i1)
		n2x, n2y, n2z := m.strideNormal(tri.i2)
		nx := w0*n0x + u*n1x + v*n2x
		ny := w0*n0y + u*n1y + v*n2y
		nz := w0*n0z + u*n1z + v*n2z
		nx, ny, nz = mulMat3(m.nMat, nx, ny, nz)
		nx, ny, nz = normalize3(nx, ny, nz)
		if finite3(nx, ny, nz) {
			hit.Normal = [3]float64{nx, ny, nz}
			hit.HasNormal = true
		}
	}
	if !hit.HasNormal {
		hit.Normal = hit.GeometricNormal
	}

	if m.obj.TextCoordFound {
		u0, v0 := m.strideUV(tri.i0)
		u1, v1 := m.strideUV(tri.i1)
		u2, v2 := m.strideUV(tri.i2)
		hit.UV = [2]float64{
			w0*u0 + u*u1 + v*u2,
			w0*v0 + u*v1 + v*v2,
		}
		hit.HasUV = true
	}

	return hit
}

// barycentric returns weights u (for vertex b) and v (for vertex c), where
// p-a = u*e1 + v*e2, e1 = b-a, e2 = c-a and n = e1 x e2.
func barycentric(px, py, pz, ax, ay, az,
	e1x, e1y, e1z, e2x, e2y, e2z float64) (float64, float64) {

	nx, ny, nz := e1y*e2z-e1z*e2y, e1z*e2x-e1x*e2z, e1x*e2y-e1y*e2x
	denom := nx*nx + ny*ny + nz*nz
	if denom == 0 {
		return 0, 0
	}
	wx, wy, wz := px-ax, py-ay, pz-az // w = p-a

	// u = n.(w x e2) / n.n
	we2X := wy*e2z - wz*e2y
	we2Y := wz*e2x - wx*e2z
	we2Z := wx*e2y - wy*e2x
	u := (nx*we2X + ny*we2Y + nz*we2Z) / denom

	// v = n.(e1 x w) / n.n
	e1wX := e1y*wz - e1z*wy
	e1wY := e1z*wx - e1x*wz
	e1wZ := e1x*wy - e1y*wx
	v := (nx*e1wX + ny*e1wY + nz*e1wZ) / denom
	return u, v
}

func crossNormalize(ax, ay, az, bx, by, bz float64) (float64, float64, float64) {
	nx := ay*bz - az*by
	ny := az*bx - ax*bz
	nz := ax*by - ay*bx
	return normalize3(nx, ny, nz)
}

func normalize3(x, y, z float64) (float64, float64, float64) {
	l := math.Hypot(math.Hypot(x, y), z)
	if l == 0 {
		return 0, 0, 0
	}
	return x / l, y / l, z / l
}

func mulMat3(a [9]float64, x, y, z float64) (float64, float64, float64) {
	return a[0]*x + a[1]*y + a[2]*z,
		a[3]*x + a[4]*y + a[5]*z,
		a[6]*x + a[7]*y + a[8]*z
}
