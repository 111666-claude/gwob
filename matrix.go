/*
Affine 4x4 matrix math used by the mesh ray picking API.

Matrices are stored row-major as 16 float64 values in the same layout as
OpenGL column vectors: a point p is transformed as M*p.
*/
package gwob

import (
	"fmt"
	"math"
)

// Matrix4 is a row-major 4x4 affine transform matrix.
type Matrix4 [16]float64

// IdentityMatrix4 returns the identity transform.
func IdentityMatrix4() Matrix4 {
	return Matrix4{
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
}

// TranslationMatrix4 builds a translation transform.
func TranslationMatrix4(x, y, z float64) Matrix4 {
	m := IdentityMatrix4()
	m[3] = x
	m[7] = y
	m[11] = z
	return m
}

// ScaleMatrix4 builds a per-axis scale transform.
func ScaleMatrix4(x, y, z float64) Matrix4 {
	return Matrix4{
		x, 0, 0, 0,
		0, y, 0, 0,
		0, 0, z, 0,
		0, 0, 0, 1,
	}
}

// RotationXMatrix4 builds a rotation of radians around the X axis.
func RotationXMatrix4(radians float64) Matrix4 {
	c, s := math.Cos(radians), math.Sin(radians)
	return Matrix4{
		1, 0, 0, 0,
		0, c, -s, 0,
		0, s, c, 0,
		0, 0, 0, 1,
	}
}

// RotationYMatrix4 builds a rotation of radians around the Y axis.
func RotationYMatrix4(radians float64) Matrix4 {
	c, s := math.Cos(radians), math.Sin(radians)
	return Matrix4{
		c, 0, s, 0,
		0, 1, 0, 0,
		-s, 0, c, 0,
		0, 0, 0, 1,
	}
}

// RotationZMatrix4 builds a rotation of radians around the Z axis.
func RotationZMatrix4(radians float64) Matrix4 {
	c, s := math.Cos(radians), math.Sin(radians)
	return Matrix4{
		c, -s, 0, 0,
		s, c, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
	}
}

// MultiplyMatrix4 returns the product a*b (apply b first, then a).
func MultiplyMatrix4(a, b Matrix4) Matrix4 {
	var r Matrix4
	for row := 0; row < 4; row++ {
		for col := 0; col < 4; col++ {
			sum := 0.0
			for k := 0; k < 4; k++ {
				sum += a[row*4+k] * b[k*4+col]
			}
			r[row*4+col] = sum
		}
	}
	return r
}

// transformPoint applies the affine matrix to a point (w = 1).
func transformPoint(m Matrix4, p [3]float64) [3]float64 {
	return [3]float64{
		m[0]*p[0] + m[1]*p[1] + m[2]*p[2] + m[3],
		m[4]*p[0] + m[5]*p[1] + m[6]*p[2] + m[7],
		m[8]*p[0] + m[9]*p[1] + m[10]*p[2] + m[11],
	}
}

// transformVector applies the linear part of the affine matrix to a vector.
func transformVector(m Matrix4, v [3]float64) [3]float64 {
	return [3]float64{
		m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
		m[4]*v[0] + m[5]*v[1] + m[6]*v[2],
		m[8]*v[0] + m[9]*v[1] + m[10]*v[2],
	}
}

// det3 returns the determinant of the upper-left 3x3 linear part of m.
func det3(m Matrix4) float64 {
	return m[0]*(m[5]*m[10]-m[6]*m[9]) -
		m[1]*(m[4]*m[10]-m[6]*m[8]) +
		m[2]*(m[4]*m[9]-m[5]*m[8])
}

// InverseMatrix4 inverts an affine (last row = 0,0,0,1) matrix. It returns an
// error for a singular (non-invertible) transform.
func InverseMatrix4(m Matrix4) (Matrix4, error) {
	const tol = 1e-15

	a := m[0]
	b := m[1]
	c := m[2]
	d := m[4]
	e := m[5]
	f := m[6]
	g := m[8]
	h := m[9]
	i := m[9+1]

	det := a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)
	if math.Abs(det) <= tol {
		return Matrix4{}, fmt.Errorf("InverseMatrix4: singular matrix determinant=%g", det)
	}
	invDet := 1 / det

	// Cofactors of the 3x3 linear part.
	A := (e*i - f*h) * invDet
	B := (c*h - b*i) * invDet
	C := (b*f - c*e) * invDet
	D := (f*g - d*i) * invDet
	E := (a*i - c*g) * invDet
	F := (c*d - a*f) * invDet
	G := (d*h - e*g) * invDet
	H := (b*g - a*h) * invDet
	I := (a*e - b*d) * invDet

	// Inverse translation: -invLinear * t.
	tx, ty, tz := m[3], m[7], m[11]
	ix := -(A*tx + B*ty + C*tz)
	iy := -(D*tx + E*ty + F*tz)
	iz := -(G*tx + H*ty + I*tz)

	return Matrix4{
		A, B, C, ix,
		D, E, F, iy,
		G, H, I, iz,
		0, 0, 0, 1,
	}, nil
}

// normalMatrix3 returns the inverse-transpose of the upper 3x3 linear part of
// the model matrix, in row-major order. inv is the already computed full
// inverse: for an affine transform the inverse-transpose linear part is the
// transpose of the inverse's linear part.
func normalMatrix3(inv Matrix4) [9]float64 {
	return [9]float64{
		inv[0], inv[4], inv[8],
		inv[1], inv[5], inv[9],
		inv[2], inv[6], inv[10],
	}
}
