/**
 * @file auxiliary.h
 * @author xbillowy
 * @brief
 * @version 0.1
 * @date 2024-08-17
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef OPTIX_TRACER_AUXILIARY_H_INCLUDED
#define OPTIX_TRACER_AUXILIARY_H_INCLUDED

#include "config.h"
#include "stdio.h"
#include "glm/glm.hpp"

#define TIGHTBBOX 0
#define DUAL_VISIABLE 1

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__forceinline__ __device__ float ndc2Pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__forceinline__ __device__ float4 transformPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float3 transformVec4x3Transpose(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[1] * p.y + matrix[2] * p.z,
		matrix[4] * p.x + matrix[5] * p.y + matrix[6] * p.z,
		matrix[8] * p.x + matrix[9] * p.y + matrix[10] * p.z,
	};
	return transformed;
}

__forceinline__ __device__ float dnormvdz(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);
	float dnormvdz = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdz;
}

__forceinline__ __device__ float3 dnormvdv(float3 v, float3 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float3 dnormvdv;
	dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
	dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
	dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
	return dnormvdv;
}

__forceinline__ __device__ float4 dnormvdv(float4 v, float4 dv)
{
	float sum2 = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
	float invsum32 = 1.0f / sqrt(sum2 * sum2 * sum2);

	float4 vdv = { v.x * dv.x, v.y * dv.y, v.z * dv.z, v.w * dv.w };
	float vdv_sum = vdv.x + vdv.y + vdv.z + vdv.w;
	float4 dnormvdv;
	dnormvdv.x = ((sum2 - v.x * v.x) * dv.x - v.x * (vdv_sum - vdv.x)) * invsum32;
	dnormvdv.y = ((sum2 - v.y * v.y) * dv.y - v.y * (vdv_sum - vdv.y)) * invsum32;
	dnormvdv.z = ((sum2 - v.z * v.z) * dv.z - v.z * (vdv_sum - vdv.z)) * invsum32;
	dnormvdv.w = ((sum2 - v.w * v.w) * dv.w - v.w * (vdv_sum - vdv.w)) * invsum32;
	return dnormvdv;
}


__forceinline__ __device__ float2&   operator/=  (float2& a, const float2& b)       {a.x /= b.x; a.y /= b.y; return a;}
__forceinline__ __device__ float2&   operator*=  (float2& a, const float2& b)       {a.x *= b.x; a.y *= b.y; return a;}
__forceinline__ __device__ float2&   operator+=  (float2& a, const float2& b)       {a.x += b.x; a.y += b.y; return a;}
__forceinline__ __device__ float2&   operator-=  (float2& a, const float2& b)       {a.x -= b.x; a.y -= b.y; return a;}
__forceinline__ __device__ float2&   operator/=  (float2& a, float b)               {a.x /= b; a.y /= b; return a;}
__forceinline__ __device__ float2&   operator*=  (float2& a, float b)               {a.x *= b; a.y *= b; return a;}
__forceinline__ __device__ float2&   operator+=  (float2& a, float b)               {a.x += b; a.y += b; return a;}
__forceinline__ __device__ float2&   operator-=  (float2& a, float b)               {a.x -= b; a.y -= b; return a;}
__forceinline__ __device__ float2    operator/   (const float2& a, const float2& b) {return make_float2(a.x / b.x, a.y / b.y);}
__forceinline__ __device__ float2    operator*   (const float2& a, const float2& b) {return make_float2(a.x * b.x, a.y * b.y);}
__forceinline__ __device__ float2    operator+   (const float2& a, const float2& b) {return make_float2(a.x + b.x, a.y + b.y);}
__forceinline__ __device__ float2    operator-   (const float2& a, const float2& b) {return make_float2(a.x - b.x, a.y - b.y);}
__forceinline__ __device__ float2    operator/   (const float2& a, float b)         {return make_float2(a.x / b, a.y / b);}
__forceinline__ __device__ float2    operator*   (const float2& a, float b)         {return make_float2(a.x * b, a.y * b);}
__forceinline__ __device__ float2    operator+   (const float2& a, float b)         {return make_float2(a.x + b, a.y + b);}
__forceinline__ __device__ float2    operator-   (const float2& a, float b)         {return make_float2(a.x - b, a.y - b);}
__forceinline__ __device__ float2    operator/   (float a, const float2& b)         {return make_float2(a / b.x, a / b.y);}
__forceinline__ __device__ float2    operator*   (float a, const float2& b)         {return make_float2(a * b.x, a * b.y);}
__forceinline__ __device__ float2    operator+   (float a, const float2& b)         {return make_float2(a + b.x, a + b.y);}
__forceinline__ __device__ float2    operator-   (float a, const float2& b)         {return make_float2(a - b.x, a - b.y);}
__forceinline__ __device__ float2    operator-   (const float2& a)                  {return make_float2(-a.x, -a.y);}
__forceinline__ __device__ float3&   operator/=  (float3& a, const float3& b)       {a.x /= b.x; a.y /= b.y; a.z /= b.z; return a;}
__forceinline__ __device__ float3&   operator*=  (float3& a, const float3& b)       {a.x *= b.x; a.y *= b.y; a.z *= b.z; return a;}
__forceinline__ __device__ float3&   operator+=  (float3& a, const float3& b)       {a.x += b.x; a.y += b.y; a.z += b.z; return a;}
__forceinline__ __device__ float3&   operator-=  (float3& a, const float3& b)       {a.x -= b.x; a.y -= b.y; a.z -= b.z; return a;}
__forceinline__ __device__ float3&   operator/=  (float3& a, float b)               {a.x /= b; a.y /= b; a.z /= b; return a;}
__forceinline__ __device__ float3&   operator*=  (float3& a, float b)               {a.x *= b; a.y *= b; a.z *= b; return a;}
__forceinline__ __device__ float3&   operator+=  (float3& a, float b)               {a.x += b; a.y += b; a.z += b; return a;}
__forceinline__ __device__ float3&   operator-=  (float3& a, float b)               {a.x -= b; a.y -= b; a.z -= b; return a;}
__forceinline__ __device__ float3    operator/   (const float3& a, const float3& b) {return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);}
__forceinline__ __device__ float3    operator*   (const float3& a, const float3& b) {return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);}
__forceinline__ __device__ float3    operator+   (const float3& a, const float3& b) {return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);}
__forceinline__ __device__ float3    operator-   (const float3& a, const float3& b) {return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);}
__forceinline__ __device__ float3    operator/   (const float3& a, float b)         {return make_float3(a.x / b, a.y / b, a.z / b);}
__forceinline__ __device__ float3    operator*   (const float3& a, float b)         {return make_float3(a.x * b, a.y * b, a.z * b);}
__forceinline__ __device__ float3    operator+   (const float3& a, float b)         {return make_float3(a.x + b, a.y + b, a.z + b);}
__forceinline__ __device__ float3    operator-   (const float3& a, float b)         {return make_float3(a.x - b, a.y - b, a.z - b);}
__forceinline__ __device__ float3    operator/   (float a, const float3& b)         {return make_float3(a / b.x, a / b.y, a / b.z);}
__forceinline__ __device__ float3    operator*   (float a, const float3& b)         {return make_float3(a * b.x, a * b.y, a * b.z);}
__forceinline__ __device__ float3    operator+   (float a, const float3& b)         {return make_float3(a + b.x, a + b.y, a + b.z);}
__forceinline__ __device__ float3    operator-   (float a, const float3& b)         {return make_float3(a - b.x, a - b.y, a - b.z);}
__forceinline__ __device__ float3    operator-   (const float3& a)                  {return make_float3(-a.x, -a.y, -a.z);}
__forceinline__ __device__ float4&   operator/=  (float4& a, const float4& b)       {a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w; return a;}
__forceinline__ __device__ float4&   operator*=  (float4& a, const float4& b)       {a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a;}
__forceinline__ __device__ float4&   operator+=  (float4& a, const float4& b)       {a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a;}
__forceinline__ __device__ float4&   operator-=  (float4& a, const float4& b)       {a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a;}
__forceinline__ __device__ float4&   operator/=  (float4& a, float b)               {a.x /= b; a.y /= b; a.z /= b; a.w /= b; return a;}
__forceinline__ __device__ float4&   operator*=  (float4& a, float b)               {a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a;}
__forceinline__ __device__ float4&   operator+=  (float4& a, float b)               {a.x += b; a.y += b; a.z += b; a.w += b; return a;}
__forceinline__ __device__ float4&   operator-=  (float4& a, float b)               {a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a;}
__forceinline__ __device__ float4    operator/   (const float4& a, const float4& b) {return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);}
__forceinline__ __device__ float4    operator*   (const float4& a, const float4& b) {return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);}
__forceinline__ __device__ float4    operator+   (const float4& a, const float4& b) {return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);}
__forceinline__ __device__ float4    operator-   (const float4& a, const float4& b) {return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);}
__forceinline__ __device__ float4    operator/   (const float4& a, float b)         {return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);}
__forceinline__ __device__ float4    operator*   (const float4& a, float b)         {return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);}
__forceinline__ __device__ float4    operator+   (const float4& a, float b)         {return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);}
__forceinline__ __device__ float4    operator-   (const float4& a, float b)         {return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);}
__forceinline__ __device__ float4    operator/   (float a, const float4& b)         {return make_float4(a / b.x, a / b.y, a / b.z, a / b.w);}
__forceinline__ __device__ float4    operator*   (float a, const float4& b)         {return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);}
__forceinline__ __device__ float4    operator+   (float a, const float4& b)         {return make_float4(a + b.x, a + b.y, a + b.z, a + b.w);}
__forceinline__ __device__ float4    operator-   (float a, const float4& b)         {return make_float4(a - b.x, a - b.y, a - b.z, a - b.w);}
__forceinline__ __device__ float4    operator-   (const float4& a)                  {return make_float4(-a.x, -a.y, -a.z, -a.w);}

__forceinline__ __device__ float3 	 cross		 (float3 a, float3 b)				{return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);}
__forceinline__ __device__ float3 	 sqrtf3 	 (float3 a)							{return make_float3(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z));}
__forceinline__ __device__ float2 	 sqrtf2 	 (float2 a)							{return make_float2(sqrtf(a.x), sqrtf(a.y));}
__forceinline__ __device__ float3 	 min	 	 (float f, float3 a)				{return make_float3(min(f, a.x), min(f, a.y), min(f, a.z));}
__forceinline__ __device__ float3 	 min	 	 (float3 a, float f)				{return make_float3(min(f, a.x), min(f, a.y), min(f, a.z));}
__forceinline__ __device__ float3  	 minf3	 	 (float f, float3 a)				{return make_float3(min(f, a.x), min(f, a.y), min(f, a.z));}
__forceinline__ __device__ float3 	 minf3  	 (float3 a, float f)				{return make_float3(min(f, a.x), min(f, a.y), min(f, a.z));}
__forceinline__ __device__ float2 	 minf2  	 (float f, float2 a)				{return make_float2(min(f, a.x), min(f, a.y));}
__forceinline__ __device__ float2 	 minf2  	 (float2 a, float f)				{return make_float2(min(f, a.x), min(f, a.y));}
__forceinline__ __device__ float3 	 max	 	 (float f, float3 a)				{return make_float3(max(f, a.x), max(f, a.y), max(f, a.z));}
__forceinline__ __device__ float3 	 max	 	 (float3 a, float f)				{return make_float3(max(f, a.x), max(f, a.y), max(f, a.z));}
__forceinline__ __device__ float3 	 maxf3  	 (float f, float3 a)				{return make_float3(max(f, a.x), max(f, a.y), max(f, a.z));}
__forceinline__ __device__ float3 	 maxf3  	 (float3 a, float f)				{return make_float3(max(f, a.x), max(f, a.y), max(f, a.z));}
__forceinline__ __device__ float2 	 maxf2  	 (float f, float2 a)				{return make_float2(max(f, a.x), max(f, a.y));}
__forceinline__ __device__ float2 	 maxf2  	 (float2 a, float f)				{return make_float2(max(f, a.x), max(f, a.y));}

__forceinline__ __device__ float 	 sumf3		 (float3 a)							{return a.x + a.y + a.z;}
__forceinline__ __device__ float 	 sumf2		 (float2 a)							{return a.x + a.y;}
__forceinline__ __device__ float 	 dot		 (const float2 a, const float2 b) 	{return a.x * b.x + a.y * b.y;}
__forceinline__ __device__ float 	 dot		 (const float3 a, const float3 b) 	{return a.x * b.x + a.y * b.y + a.z * b.z;}
__forceinline__ __device__ float 	 dot		 (const float4 a, const float4 b) 	{return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;}
__forceinline__ __device__ float 	 norm		 (const float2 a) 					{return sqrtf(dot(a, a));}
__forceinline__ __device__ float 	 norm		 (const float3 a) 					{return sqrtf(dot(a, a));}
__forceinline__ __device__ float 	 norm		 (const float4 a) 					{return sqrtf(dot(a, a));}

__forceinline__ __device__ float3 matmul33x3(const float3* mat, const float3 vec)
{
    return make_float3(dot(mat[0], vec), dot(mat[1], vec), dot(mat[2], vec));
}
__forceinline__ __device__ float4 matmul44x4(const float4* mat, const float4 vec)
{
    return make_float4(dot(mat[0], vec), dot(mat[1], vec), dot(mat[2], vec), dot(mat[3], vec));
}
__forceinline__ __device__ void matmul33x33(const float3* mat1, const float3* mat2, float3* mat)
{
	mat[0] = make_float3(mat1[0].x * mat2[0].x + mat1[0].y * mat2[1].x + mat1[0].z * mat2[2].x,
						 mat1[0].x * mat2[0].y + mat1[0].y * mat2[1].y + mat1[0].z * mat2[2].y,
						 mat1[0].x * mat2[0].z + mat1[0].y * mat2[1].z + mat1[0].z * mat2[2].z);
	mat[1] = make_float3(mat1[1].x * mat2[0].x + mat1[1].y * mat2[1].x + mat1[1].z * mat2[2].x,
						 mat1[1].x * mat2[0].y + mat1[1].y * mat2[1].y + mat1[1].z * mat2[2].y,
						 mat1[1].x * mat2[0].z + mat1[1].y * mat2[1].z + mat1[1].z * mat2[2].z);
	mat[2] = make_float3(mat1[2].x * mat2[0].x + mat1[2].y * mat2[1].x + mat1[2].z * mat2[2].x,
						 mat1[2].x * mat2[0].y + mat1[2].y * mat2[1].y + mat1[2].z * mat2[2].y,
						 mat1[2].x * mat2[0].z + mat1[2].y * mat2[1].z + mat1[2].z * mat2[2].z);
}

__forceinline__ __device__ float3 correct_normal(const float3& N, const float3& I)
{
	return dot(I, N) < 0 ? N : -N;
}


__forceinline__ __device__ bool in_frustum(int idx,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool prefiltered,
	float3& p_view)
{
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	// float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	// float p_w = 1.0f / (p_hom.w + 0.0000001f);
	// float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		if (prefiltered)
		{
			printf("Point is filtered although prefiltered is set. This shouldn't happen!");
			__trap();
		}
		return false;
	}
	return true;
}


// Adopt from gsplat: https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/csrc/forward.cu
inline __device__ glm::mat3 quat_to_rotmat(const glm::vec4 quat) {
	// quat to rotation matrix
	float s = rsqrtf(
		quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
	);
	float w = quat.x * s;
	float x = quat.y * s;
	float y = quat.z * s;
	float z = quat.w * s;

	// glm matrices are column-major
	return glm::mat3(
		1.f - 2.f * (y * y + z * z),
		2.f * (x * y + w * z),
		2.f * (x * z - w * y),
		2.f * (x * y - w * z),
		1.f - 2.f * (x * x + z * z),
		2.f * (y * z + w * x),
		2.f * (x * z + w * y),
		2.f * (y * z - w * x),
		1.f - 2.f * (x * x + y * y)
	);
}

	// quat to rotation matrix
inline __device__ glm::mat3 quat_to_rotmat_transpose(const glm::vec4 quat) {
	float s = rsqrtf(
		quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
	);
	float w = quat.x * s;
	float x = quat.y * s;
	float y = quat.z * s;
	float z = quat.w * s;

	// This is a transpose version of the above function
	// glm matrices are column-major
	return glm::mat3(
		1.f - 2.f * (y * y + z * z),
		2.f * (x * y - w * z),
		2.f * (x * z + w * y),
		2.f * (x * y + w * z),
		1.f - 2.f * (x * x + z * z),
		2.f * (y * z - w * x),
		2.f * (x * z - w * y),
		2.f * (y * z + w * x),
		1.f - 2.f * (x * x + y * y)
	);
}


// Adopt from gsplat: https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/csrc/forward.cu
inline __device__ void quat_to_rotmat(const float4& quat, float3* R) {
	// quat to rotation matrix
	float s = rsqrtf(
		quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
	);
	float w = quat.x * s;
	float x = quat.y * s;
	float y = quat.z * s;
	float z = quat.w * s;

	// column-major?
	R[0] = make_float3(
		1.f - 2.f * (y * y + z * z),
		2.f * (x * y - w * z),
		2.f * (x * z + w * y)
	);
	R[1] = make_float3(
		2.f * (x * y + w * z),
		1.f - 2.f * (x * x + z * z),
		2.f * (y * z - w * x)
	);
	R[2] = make_float3(
		2.f * (x * z - w * y),
		2.f * (y * z + w * x),
		1.f - 2.f * (x * x + y * y)
	);
}

inline __device__ float3 quat_to_rotmat_transpose(const float4& quat, float3* R) {
	// quat to rotation matrix
	float s = rsqrtf(
		quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
	);
	float w = quat.x * s;
	float x = quat.y * s;
	float y = quat.z * s;
	float z = quat.w * s;

	// This is a transpose version of the above function
	R[0] = make_float3(
		1.f - 2.f * (y * y + z * z),
		2.f * (x * y + w * z),
		2.f * (x * z - w * y)
	);
	R[1] = make_float3(
		2.f * (x * y - w * z),
		1.f - 2.f * (x * x + z * z),
		2.f * (y * z + w * x)
	);
	R[2] = make_float3(
		2.f * (x * z + w * y),
		2.f * (y * z - w * x),
		1.f - 2.f * (x * x + y * y)
	);
}


inline __device__ glm::vec4
quat_to_rotmat_vjp(const glm::vec4 quat, const glm::mat3 v_R) {
	float s = rsqrtf(
		quat.w * quat.w + quat.x * quat.x + quat.y * quat.y + quat.z * quat.z
	);
	float w = quat.x * s;
	float x = quat.y * s;
	float y = quat.z * s;
	float z = quat.w * s;

	glm::vec4 v_quat;
	// v_R is COLUMN MAJOR
	// w element stored in x field
	v_quat.x =
		2.f * (
				  // v_quat.w = 2.f * (
				  x * (v_R[1][2] - v_R[2][1]) + y * (v_R[2][0] - v_R[0][2]) +
				  z * (v_R[0][1] - v_R[1][0])
			  );
	// x element in y field
	v_quat.y =
		2.f *
		(
			// v_quat.x = 2.f * (
			-2.f * x * (v_R[1][1] + v_R[2][2]) + y * (v_R[0][1] + v_R[1][0]) +
			z * (v_R[0][2] + v_R[2][0]) + w * (v_R[1][2] - v_R[2][1])
		);
	// y element in z field
	v_quat.z =
		2.f *
		(
			// v_quat.y = 2.f * (
			x * (v_R[0][1] + v_R[1][0]) - 2.f * y * (v_R[0][0] + v_R[2][2]) +
			z * (v_R[1][2] + v_R[2][1]) + w * (v_R[2][0] - v_R[0][2])
		);
	// z element in w field
	v_quat.w =
		2.f *
		(
			// v_quat.z = 2.f * (
			x * (v_R[0][2] + v_R[2][0]) + y * (v_R[1][2] + v_R[2][1]) -
			2.f * z * (v_R[0][0] + v_R[1][1]) + w * (v_R[0][1] - v_R[1][0])
		);
	return v_quat;
}


inline __device__ glm::mat3
scale_to_mat(const glm::vec2 scale, const float glob_scale) {
	glm::mat3 S = glm::mat3(1.f);
	S[0][0] = glob_scale * scale.x;
	S[1][1] = glob_scale * scale.y;
	// S[2][2] = glob_scale * scale.z;
	return S;
}

inline __device__ glm::mat3
scale_to_mat_inverse(const glm::vec2 scale, const float glob_scale) {
	glm::mat3 S = glm::mat3(1.f);
	S[0][0] = 1.f / (glob_scale * scale.x);
	S[1][1] = 1.f / (glob_scale * scale.y);
	// S[2][2] = 1.f / (glob_scale * scale.z);
	return S;
}


inline __device__ void
scale_to_mat(const float2& scale, const float glob_scale, float3* S) {
	S[0] = make_float3(glob_scale * scale.x, 0.f, 0.f);
	S[1] = make_float3(0.f, glob_scale * scale.y, 0.f);
	S[2] = make_float3(0.f, 0.f, 1.f);
}

inline __device__ void
scale_to_mat_inverse(const float2& scale, const float glob_scale, float3* S) {
	S[0] = make_float3(1.f / (glob_scale * scale.x), 0.f, 0.f);
	S[1] = make_float3(0.f, 1.f / (glob_scale * scale.y), 0.f);
	S[2] = make_float3(0.f, 0.f, 1.f);
}


#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret); \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

#endif
