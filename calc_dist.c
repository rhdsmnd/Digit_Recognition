/*
 * PROJ1-1: YOUR TASK A CODE HERE
 *
 * You MUST implement the calc_min_dist() function in this file.
 *
 * You do not need to implement/use the swap(), flip_horizontal(), transpose(), or rotate_ccw_90()
 * functions, but you may find them useful. Feel free to define additional helper functions.
 */

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <nmmintrin.h>
#include "digit_rec.h"
#include "utils.h"
#include "limits.h"


/* Swaps the values pointed to by the pointers X and Y. */
void swap(float *x, float *y) {
	float temp=  *y;
	*y = *x;
	*x = temp;
}
/* Flips the elements of a square array ARR across the y-axis. */
void flip_horizontal(float *arr, int dim) {
  int h = dim;
  int w = dim/2;
  #pragma omp parallel
  {
  #pragma omp for
		for(int i =0; i< h * w; i++){
			int x = i % w;
      int y = i / w;
			swap(&(arr[h*y+x]), &(arr[h*y+dim-1-x]));
		}
  }

}
/* Transposes the square array ARR. */
void transpose(float *arr, int dim) {
	/* Optional function */
  #pragma omp parallel
  {

  #pragma omp for
	for (int y = 0; y<dim-2; y +=1){
        for(int x=y+1; x<dim-2; x+=1){
		swap(&arr[dim*y+x], &arr[dim*x + y]);
        }
    }
  }
}

/* Rotates the square array ARR by 90 degrees counterclockwise. */
void rotate_ccw_90(float *arr, int dim) {
	flip_horizontal(arr, dim);
	transpose(arr, dim);
}

//creates a new array containing elements of coppied array
float* copy(float *arr, int dim){
	float * copy = (float*) malloc(dim*sizeof(float));
	#pragma omp parallel
	{
	#pragma omp for
	for(int i = 0; i<dim; i++){
		 copy[i] = arr[i];
	}
	}
	return copy;
}

//Square
float square(float x){
	return x * x;
}

float min2(float a, float b) {
	if (a < b) {
	    return a;
	}
	return b;
}

float findMin(float a, float b, float c, float d, float e) {
	float min = min2(a, b);
	min = min2(min, c);
	min = min2(min, d);
	min = min2(min, e);
	return min;
}

/* Returns the squared Euclidean distance between TEMPLATE and IMAGE. The size of IMAGE
 * is I_WIDTH * I_HEIGHT, while TEMPLATE is square with side length T_WIDTH. The template
 * image should be flipped, rotated, and translated across IMAGE.
 */

//todo: vectorize the add and mul in inner loop, find a better way to write back and collect min of list
float calc_min_dist(float *image, int i_width, int i_height,
        float *template, int t_width) {
    
    float min = FLT_MAX;
	/** keep track of all 8 orientations at once in order to
	 *  make use of temporal locality/register blocking.
	 *  abbreviations: n = normal, cw and ccw = (counter)clockwise,
	 *  ud = upside down, f = flipped. */
	#pragma omp parallel for
	for(int i = 0; i<(i_width - t_width +1)*(i_height - t_width+1); i++){
		int dx = i %(i_width - t_width +1);;
		int dy = i /(i_width - t_width +1);
		float n = 0;
		float fn = 0;
		float cw = 0;
		float fcw = 0;
		float ccw = 0;
		float fccw = 0;
		float ud = 0;
		float fud = 0;
		for(int y = 0; y < t_width; y += 1){
			for(int x = 0; x < t_width; x+=1){
				n += square(image[(y + dy)* i_width + x + dx] - template[y * t_width + x]);
				fn +=square(image[(y+dy)*i_width+x+dx] - template[y*t_width + (t_width - 1 -x)]);
				cw +=square(image[(y+dy)*i_width+x+dx] - template[t_width - 1 - y
										  + (t_width * x)]);
				fcw +=square(image[(y+dy)*i_width+x+dx] - template[x * t_width + y]);
				ccw +=square(image[(y+dy)*i_width+x+dx] - template[t_width - 1 + x*t_width - y]);
				fccw += square(image[(y + dy) * i_width + x + dx]
					       - template[t_width * t_width - 1 - y - x * t_width]);
				ud +=square(image[(y+dy)*i_width+x+dx] - template[(t_width *t_width) - 1 - x
											- y * t_width]);
				fud +=square(image[(y+dy)*i_width+x+dx] - template[(t_width - 1)*t_width + x
											- t_width * y]);
			}
		}
                #pragma omp critical 
		min = findMin(min, n, fn, cw, fcw);
		#pragma omp critical
		min = findMin(min, ccw, fccw, ud, fud);
	}
	return min;
}

