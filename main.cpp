//author: Lakshwin
//date: 24th July 2023 @ 1500
//location: Paris, Cite universite

#include <iostream>
#include <math.h>
#include <string>
#include <vector>

using namespace std;


struct doubleVector (){
	
	doubleVector(int size){
		int size = size;
		vector<int> src_neighbors (size);
		vector<int> tar_neighbors (size); // size = 8 because it holds neighbors in each of the 8 different directions of a cell
										  //
		// initialize to -1 (implying the neighbor does not exist)
		for (int i = 0; i<size; ++i){
			src_neighbors[i] = -1;
			tar_neighbors[i] = -1;
		}
	}
};

int roll_dice(int n_kinds){

	// rolls a dice based on the seed initialized in main

	if(n_kinds <0) throw runtime_error("cell types must be a positive value\n");
	//prng generator
	int val =  rand() % n_kinds;
	return val;
}

vector<vector<int>> get_individual(int size, int n_kinds){

	//creates a random 2d individual

	if (size <0) throw runtime_error("negative size not allowed\n");
	vector<vector<int>> v1;

	// initialize random elements based on number of cell kinds
	// note: i = row and j = column
	//
	//
	for(int i = 0; i <int(size); ++i){
		vector<int> col; 

		for (int j = 0 ; j<int(size); ++j){
			int val = roll_dice(n_kinds);
			col.push_back(val);
		}
		v1.push_back(col);
	}	

	return v1;
} 

void print_individual(vector<vector<int>> &v1){

	//prints elements of a 2d individual onto std output
	//
	if (v1.size() ==0) throw runtime_error("source matrix is empty\n");

	for (int i = 0; i<int(v1.size()); ++i){
		vector<int> all_cols = v1[i]; 

		for (int j = 0; j<int(all_cols.size()); ++j){
			cout<<all_cols[j]<<" ";
		}
		cout<<"\n";
	}
}

double fitness(vector<vector<int>> &source, vector<vector<int>> &target){

	//implements eucledian, l2 loss between src and target matrices of the same size

	if (source.size() ==0 || target.size()==0) throw runtime_error("source/target matrices must have size >1\n"); 

	if (source.size() != target.size()) throw runtime_error("source and target matrices must be of the same size\n");

	double error = 0.0;


	// temp placeholders (initialized only once, hence outside the for loop)
	vector<int> src_temp;
	vector<int> tar_temp;


	for (int i = 0; i<int(source.size()); ++i){
		src_temp = source[i];
		tar_temp = target[i];

		for (int j = 0; j<int(source.size()); ++j){
			error += pow((src_temp[j] - tar_temp[j]), 2); 
		}
	}

	return error/double(pow(source.size(), 2));

}

doubleVector get_neighbors(int i, int j, int size, vector<vector<int>> &src, vector<vector<int>> &tar){

	// size is pre-conditioned

	if(i<0 || j <0) throw runtime_error("indices must be positive\n");
	if (i>size || j>size) throw runtime_error("indices must be less than size of the matrix\n");

	doubleVector v1(8); // 8 directions, order:  n, s, w, e, nw, ne, sw, se
	
	//check if within bounds in each direction
	//
	//north:
	if (i-1 >= 0){
		v1.src_neighbors[0] = src[i-1][j];
		v1.tar_neighbors[0] = tar[i-1][j];
	} 

	//south:
	if (i+1 < size){
		v1.src_neighbors[1] = src[i+1][j];
		v1.tar_neighbors[1] = tar[i+1][j];
	}

	//west:
	if (j-1 >=0){
		v1.src_neighbors[2] = src[i][j-1];
		v1.tar_neighbors[2] = tar[i][j-1];
	}

	//east:
	if (j+1 <size) {
		v1.src_neighbors[3] = src[i][j+1];
		v1.tar_neighbors[3] = tar[i][j+1];
	}

	//north-west:
	if (i-1 >=0 && j-1 >=0) {
		v1.src_neighbors[4] = src[i-1][j-1];
		v1.tar_neighbors[4] = tar[i-1][j-1];

	}

	//north-east:
	if (i-1 >=0 && j+1 <size) {
		v1.src_neighbors[5] = src[i-1][j+1];
		v1.tar_neighbors[5] = tar[i-1][j+1];

	}

	//south-west:
	if (i+1 <size && j-1 >=0) {
		v1.src_neighbors[6] = src[i+1][j-1];
		v1.tar_neighbors[6] = tar[i+1][j-1];

	}

	//south-east:
	if (i+1 <size && j+1 <size) {
		v1.src_neighbors[7] = src[i+1][j+1];
		v1.tar_neighbors[7] = tar[i+1][j+1];

	}

	return v1;
}

vector<double> compare_neighbors(doubleVector &neighbors){

	//m and n are preconditioned, src and tar are as well
	// pre-condition for neighbors array:
	int count = 0;
	for (int i =0; i<neighbors.size; ++i){
		if (neighbors.src_neighbor[i] >0 || neighbors.tar_neighbor[i] > 0) count ++;
	}

	if (count == 0) throw runtime_error("Neighborhood array is empty\n");

	//function body
	//
	
	vector<double> stressed_positions(8);

	for (int i = 0; i<int(neighbors.size); ++i){ //cycle through each of the 8 directions and get neighbors in both the src and tar matrix
		if (neighbors.src_neighbors[i] == -1) stressed_positions[i] = -1.0;

		else{
			stressed_positions[i] = double(abs(neighbors.src_neighbors[i] - neighbors.tar_neighbors[i]));
		}	
	}

	return stressed_positions;

}

double summate(vector<double> &temp){
	double sum = 0.0;
	int count = 0;
	for (int x: temp){
		if (x != -1.0){
			sum += x;
			count += 1;
		}
	}

	return sum/double(count);
}

void calculate_stress(vector<vector<int>> &src, vector<vector<int>> &tar, vector<vector<vector<double>>> &positional_stress, vector<vector<double>> &stress){

	//get the unhappiness level of a cell-position

	// src and tar will be pre-conditioned in apply_competency. Positional stress does not require pre-conditioning 
	
	doubleVector neighbors;

	int size = int(src.size());

	for(int i = 0; i < size; ++i){
		stress_temp = stress[i];

		for (int j = 0; j < size; ++j){
			// for each position check stress in each of the 8 diff directions. 
			// if the direction does not exist then set it to 0.
			// make sure to count the number of directions which exist (for normalization)
			neighbors = get_neighbors(i, j, size);

			vector<double> temp = compare_neighbors(neighbors);
			positional_stress[i][j] = temp;

			stress[i][j] = summate(temp); // get total stress based on all 8 positions, ignore -1's;
		}
	}	
}

void swap(vector<vector<int>> &src, int cv){

	// src and cv will be pre-conditioned in apply_competency 
		
}

void initialize_3d(int size, vector<vector<vector<double>>> &positional_stress){

	// initialize a 3d matrix with zeros; this serves as the positional_stress matrix;

	for (int i = 0; i<size; ++i){

		vector<vector<double>> b(size);
		for (int j = 0; j<size; ++j){

			vector<double> h(size);
			for (int k = 0; j< 8; ++k){

				h.push_back(0.0);
			}
			b.push_back(h);
		} 
		positional_stress.push_back(b)
	}

}

void initialize_2d(int size, vector<vector<double>> &stress){
	for (int i = 0; i<size; ++i){
		vector<double> temp;
		for (int j = 0; j<size; ++j){
			temp.push_back(0.0);
		}

		stress.push_back(temp);
	}
}

void apply_competency(vector<vector<int>> &src, vector<vector<int>> &tar, int cv){

	// reshuffles the src matrix based on the competency value 
	
	if (cv <0) throw runtime_error("Competency value must be positive\n"); // basic checks
	if (src.size()<0) throw runtime_error("src matrix must have some elements\n"); //basic checks
	if (src.size() != tar.size()) throw runtime_error("src and target matrices must be of the same size\n");
	
	// step1: check stress ->
	// assumption: the set of cells in src exist in the target. Mutation does not change the value of a cell, it only dislocates it
	// checks the unhappiness level of a position
	// unhappiness is based on the cell-position rather than the value it contains (if a position is surrounded by the correct neighbors it must have the correct value)
	// stress will be calculated based on neighborhood positions in the tar and will be normalized
	// step2: swap -> the cell with the greatest stress will try moving with one of its stress inducing neighbors (chosen randomly)
	// the swap sticks if the stress of the position reduces
	
	vector<vector<vector<double>>> positional_stress; // every cell (i,j) has a 8 dimensional vector indicating in which directions it is stressed
	initialize_3d(positional_stress); // initialize 3d matrix with 0's
									  //
	// we also need the total stress in each (i,j) to determine which position should undergo swapping
	
	vector<vector<double>> stress;	
	initialize_2d(stress); // set elements to 0.0

	calculate_stress(src, tar, positional_stress, stress); // stress will be a matrix: each position will carry a stress value
	swap(src, stress, cv);
}



int main(){

	//assumption: 2d matrices are square

	int seed = 10;
	int size = 5;
	int n_kinds = 2; 

	srand(seed);

	try{

	vector<vector<int>> v1 = get_individual(size, n_kinds);
	print_individual(v1);

	return 0;	
	}

	catch(runtime_error &e){
		cerr<<"RuntimeError: "<<e.what()<<"\n";
		return 1;
	}

	catch(...){
		cerr<<"UnknownError\n";
		return 2;
	}
}
