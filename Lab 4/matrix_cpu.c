// Matrix addition, CPU version
// gcc matrix_cpu.c -o matrix_cpu -std=c99

#include <stdio.h>
#include <stdlib.h>
#include "milli.h"

void add_matrix(float *a, float *b, float *c, int N)
{
	int index;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			index = i + j*N;
			c[index] = a[index] + b[index];
		}
}

int main()
{
	const int N = 16;

	float *a;
	float *b;
	float *c;
	a = (float *)malloc((N*N)*sizeof(float));
	b = (float *)malloc((N*N)*sizeof(float));
	c = (float *)malloc((N*N)*sizeof(float));

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}

	ResetMilli();
	add_matrix(a, b, c, N);
	double time = (double)GetMicroseconds()/1000.0;

/*
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%0.2f ", c[i+j*N]);
		}
		printf("\n");
	}
*/
	printf("\n*----------- BENCHMARKING -----------*");
	printf("\n\nThe filtering was finished in %f microseconds. \n\n", time );

}
