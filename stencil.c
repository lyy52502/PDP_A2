#include "stencil.h"
#include<mpi.h>
#include<stdio.h>
#include<stdint.h>
#include<math.h>
#include<string.h>


void allocate_arr(int n, int k, int size, int* first_k, int* last_k) {
    int q = n / size;
    int remainder = n % size;

    if (remainder == 0) {
        *first_k = k * q;
        *last_k = (k + 1) * q - 1;
    } else {
        if (k < remainder) {
            *first_k = k * (q + 1);
            *last_k = (k + 1) * (q + 1) - 1;
        } else {
            *first_k = remainder * (q + 1) + (k - remainder) * q;
            *last_k = *first_k + q - 1;
        }
    }
}

void parallel_stencil(double *input, double *global_output, int num_values, int num_steps, int rank, int size, const double *STENCIL, int STENCIL_WIDTH, int EXTENT){
	int first_k, last_k;
    allocate_arr(num_values, rank, size, &first_k, &last_k);

    int local_N = last_k - first_k + 1;
    double *local_input = malloc((local_N + 2 * EXTENT) * sizeof(double));
    double *local_output = malloc((local_N + 2 * EXTENT) * sizeof(double));
	int *counts = malloc(size * sizeof(int));
	int *displs = malloc(size * sizeof(int));
	int sum = 0;
	for (int i = 0; i < size; ++i) {
		int f, l;
		allocate_arr(num_values, i, size, &f, &l);
		counts[i] = l - f + 1;
		displs[i] = sum;
		sum += counts[i];
	}
	MPI_Scatterv(input, counts, displs, MPI_DOUBLE,
		local_input + EXTENT, local_N, MPI_DOUBLE,
		0, MPI_COMM_WORLD);
	
	int left_rank = (rank == 0) ? size - 1 : rank - 1;
    int right_rank = (rank == size - 1) ? 0 : rank + 1;

	 // Perform stencil operations for the specified number of steps
	 for (int step = 0; step < num_steps; step++) {
        // Get boundary values from neighbors
        
		// input=[1,2,3,4,5,6,7,8], rank0=[1,2,3,4], rank1=[5,6,7,8]                                                           
		// send the two left elements[1,2] to left neighbor, and receive the last two element[7,8] from left ne$
        MPI_Sendrecv(local_input + EXTENT, EXTENT, MPI_DOUBLE, left_rank, 0,
			local_input + local_N + EXTENT, EXTENT, MPI_DOUBLE, right_rank, 0,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
        // send the two right elements[3,4] to right neighbor
        MPI_Sendrecv(local_input + local_N, EXTENT, MPI_DOUBLE, right_rank, 1,
			local_input, EXTENT, MPI_DOUBLE, left_rank, 1,
			MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
        
        // Apply stencil
		for (int i = 0; i < local_N; i++) {
            double result = 0;
            for (int j = 0; j < STENCIL_WIDTH; j++) {
                result += STENCIL[j] * local_input[i + j];
            }
            local_output[i + EXTENT] = result;
		}
        
        // Swap input and output 
        double *temp = local_input;
        local_input = local_output;
        local_output = temp;
    }
    
    // Gather results back to root process
    MPI_Gatherv(local_input + EXTENT, local_N, MPI_DOUBLE,
		global_output, counts, displs, MPI_DOUBLE,
		0, MPI_COMM_WORLD);
    

    free(local_input);
    free(local_output);
	free(counts);
	free(displs);
}


int main(int argc, char **argv) {
	if (4 != argc) {
		printf("Usage: stencil input_file output_file number_of_applications\n");
		return 1;
	}

	MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

	char *input_name = argv[1];
	char *output_name = argv[2];
	int num_steps = atoi(argv[3]);

	// Read input file
	double *input=NULL;
	int num_values;
	if (0 > (num_values = read_input(input_name, &input))) {
		return 2;
	}

	// Stencil values
	double h = 2.0*PI/num_values;
	const int STENCIL_WIDTH = 5;
	const int EXTENT = STENCIL_WIDTH/2;
	const double STENCIL[] = {1.0/(12*h), -8.0/(12*h), 0.0, 8.0/(12*h), -1.0/(12*h)};

	// Allocate data for result
	double *output = NULL;
	if (rank == 0) {
		output = malloc(num_values * sizeof(double));
		if (output == NULL) {
			perror("Couldn't allocate memory for output");
			free(input);
			MPI_Finalize();
			return 2;
		}
	}
	
	// Start timer
	MPI_Barrier(MPI_COMM_WORLD);
	double start = MPI_Wtime();
	parallel_stencil(input, output, num_values, num_steps, rank, size, STENCIL, STENCIL_WIDTH, EXTENT);
	// Stop timer
	double local_time = MPI_Wtime() - start;
	double max_time;
	MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (rank==0){
		printf("%f\n", max_time);
	}
#ifdef PRODUCE_OUTPUT_FILE
	if (rank == 0) {
		if (0 != write_output(output_name, output, num_values)) {
			free(output);
			free(input);
			MPI_Finalize();
			return 2;
		}
	}
#endif
// Clean up
	if (rank == 0) {
		free(output);
	}
	free(input);

	MPI_Finalize();
	return 0;
}


int read_input(const char *file_name, double **values) {
	FILE *file;
	if (NULL == (file = fopen(file_name, "r"))) {
		perror("Couldn't open input file");
		return -1;
	}
	int num_values;
	if (EOF == fscanf(file, "%d", &num_values)) {
		perror("Couldn't read element count from input file");
		return -1;
	}
	if (NULL == (*values = malloc(num_values * sizeof(double)))) {
		perror("Couldn't allocate memory for input");
		return -1;
	}
	for (int i=0; i<num_values; i++) {
		if (EOF == fscanf(file, "%lf", &((*values)[i]))) {
			perror("Couldn't read elements from input file");
			return -1;
		}
	}
	if (0 != fclose(file)){
		perror("Warning: couldn't close input file");
	}
	return num_values;
}


int write_output(const char *file_name, const double *output, int num_values) {
	FILE *file;
	if (NULL == (file = fopen(file_name, "w"))) {
		perror("Couldn't open output file");
		return -1;
	}
	for (int i = 0; i < num_values; i++) {
		if (0 > fprintf(file, "%.4f ", output[i])) {
			perror("Couldn't write to output file");
		}
	}
	if (0 > fprintf(file, "\n")) {
		perror("Couldn't write to output file");
	}
	if (0 != fclose(file)) {
		perror("Warning: couldn't close output file");
	}
	return 0;

	
}