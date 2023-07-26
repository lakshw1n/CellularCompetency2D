//author: Lakshwin
//date: 24th July 2023 @ 1500
//location: Paris, Cite universite

#include <iostream>
#include <math.h>
#include <string>
#include <vector>

using namespace std;

void print_positional_stress(int size, int n_directions, vector<vector<vector<double>>> &ps);

void print_totalStress(vector<vector<double>> &stress);

class doubleVector{
	public:
		int size;
		vector<int> src_neighbors; 
		vector<int> tar_neighbors;

		doubleVector(int sz){
			size = sz;
			for (int i = 0; i<size; ++i){
				src_neighbors.push_back(-1);
				tar_neighbors.push_back(-1);
			}
		}

		void print_neighbors(void);
};

void doubleVector::print_neighbors(void){
	//prints src and target neighbors;
	cout<<"src_neighbors: ";
	for (int i =0; i< size; ++i){
		cout<<src_neighbors[i]<<", ";
	}

	cout<<"\ntar_neighbors: ";
	for (int i =0; i< size; ++i){
		cout<<tar_neighbors[i]<<", ";
	}
	cout<<"\n==========\n";
}


int roll_dice(int n_kinds){

	// rolls a dice based on the seed initialized in main

	if(n_kinds <0) throw runtime_error("cell types must be a positive value\n");
	//prng generator
	int val =  rand() % n_kinds;
	return val;
}

vector<vector<int>> get_individual(vector<vector<int>> &target){

	//creates a random 2d individual by scrambling the target matrix
	//pick two locations in the matrix to interchange, repeat for a number of times based on the size of the matrix

	if (target.size() <=0) throw runtime_error("Empty target matrix\n");
	vector<vector<int>> v1;

	vector<vector<int>> src = target; // create a copy of target for scambling	
	
	int count = 0;
	int idx1_i = -1, idx1_j = -1;
	int idx2_i = -1, idx2_j = -1;

	int size = int(src.size());

	int temp = -100;

	while (count<= size*size){	
		idx1_i = roll_dice(size-1); // get an index from [0, array_size)
		idx1_j = roll_dice(size-1);	

		idx2_i = roll_dice(size-1);
		idx2_j = roll_dice(size-1);

		if ((idx1_i == idx2_i) && (idx1_j == idx2_j)) continue;
		
		temp = src[idx1_i][idx1_j];
		src[idx1_i][idx1_j] = src[idx2_i][idx2_j];
		src[idx2_i][idx2_j] = temp;
		
		count ++;
	}
		
	return src;
} 

void print_individual(vector<vector<int>> &v1){

	//prints elements of a 2d individual onto std output
	//
	if (v1.size() ==0) throw runtime_error("source matrix is empty\n");

	for (int i = 0; i<int(v1.size()); ++i){
		for (int j = 0; j<int(v1.size()); ++j){
			cout<<v1[i][j]<<" ";
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

	//check neighborhood values in src and target, if they aren't similar, then calculate stress as their absolute difference. 

	//m and n are preconditioned, src and tar are as well
	// pre-condition for neighbors array:
	int count = 0;
	for (int i =0; i<neighbors.size; ++i){
		if (neighbors.src_neighbors[i] >0 || neighbors.tar_neighbors[i] > 0) count ++;
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

	// src and tar will be pre-conditioned in apply_competency. Positional stress & stress do not require pre-conditioning.
	
	doubleVector neighbors(8);

	int size = int(src.size());

	for(int i = 0; i < size; ++i){
		for (int j = 0; j < size; ++j){
			// for each position check stress in each of the 8 diff directions. 
			// if the direction does not exist then set it to 0.
			// make sure to count the number of directions which exist (for normalization)
			neighbors = get_neighbors(i, j, size, src, tar);	

			vector<double> temp = compare_neighbors(neighbors);
			positional_stress[i][j] = temp;

			stress[i][j] = summate(temp); // get total stress based on all 8 positions, ignore -1's;
		}
	}	
	/* print_totalStress(stress); */
}

void swap(vector<vector<int>> &src, vector<vector<vector<double>>> &positional_stress, vector<vector<double>> &stress, int cv){

	// src and cv will be pre-conditioned in apply_competency 
	// stress matrix does not require pre-conditioning
	
	// repeat:
	// 	pick the index_position with the highest stress
	//  get the positional stress values of the specific index
	//  pick a random neighbor to swap with (make sure the neighbor exists, i.e, skip -1's)
	//  check if the stress of the position improves, if yes, keep, else pick another neighbor to swap with
	//  increment swap counter
	//  recheck for the most stressed position, and repeat.
	//  repeat until swap counter exceeds competency_value: cv, or if stress levels are 0 in all positions 
	//
	//  There seems to be no point to stress sharing, because sharing stress does not change the tendency of each cell-position to swap until its stress is low.
	//  Also, Swaps cannot be with the max stress inducing neighbor, it has to be random so that cyclic behviour does not result (simply gets stuck swapping between two positions)
		
}

void initialize_3d(int size, int n_directions, vector<vector<vector<double>>> &positional_stress){

	// initialize a 3d matrix with zeros; this serves as the positional_stress matrix;

	for (int i = 0; i<size; ++i){
		vector<vector<double>> b;
		for (int j = 0; j<size; ++j){

			vector<double> h;
			for (int k = 0; k< n_directions; ++k){
				h.push_back(0.0);
			}
			b.push_back(h);
		} 
		positional_stress.push_back(b);
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

void print_positional_stress(int size, int n_directions, vector<vector<vector<double>>> &ps){
	//prints a 3d matrix (kind of)
	//pre-conditioning irrelevent
	for (int i = 0; i<size; ++i){
		for (int j = 0; j<size; ++j){
			cout<<"Element "<< i<<","<<j<<" neighbors: ";
			for (int k = 0; k<n_directions; ++k){
				cout<<ps[i][j][k]<<", ";
			}
			cout<<"\n";
		}
	}
}

void print_totalStress(vector<vector<double>> &stress){
	//prints a 2d matrix of type double
	
	for (int i = 0; i< int(stress.size()); ++i){
		for (int j = 0; j< int(stress.size()); ++j){
			cout<<stress[i][j]<<", ";
		}
		cout<<"\n";
	}
}

void apply_competency(vector<vector<int>> &src, vector<vector<int>> &tar, int cv, int n_directions){

	// reshuffles the src matrix based on the competency value 
	
	if (cv <0) throw runtime_error("Competency value must be positive\n"); // basic checks
	if (src.size()<0) throw runtime_error("src matrix must have some elements\n"); //basic checks
	if (src.size() != tar.size()) throw runtime_error("src and target matrices must be of the same size\n");

	int size = int(src.size());
	
	// step1: check stress ->
	// assumption: the set of cells in src exist in the target. Mutation does not change the value of a cell, it only dislocates it
	// checks the unhappiness level of a position
	// unhappiness is based on the cell-position rather than the value it contains (if a position is surrounded by the correct neighbors it must have the correct value)
	// stress will be calculated based on neighborhood positions in the tar and will be normalized
	// step2: swap -> the cell with the greatest stress will try moving with one of its stress inducing neighbors (chosen randomly)
	// the swap sticks if the stress of the position reduces
	
	vector<vector<vector<double>>> positional_stress; // every cell (i,j) has a 8 dimensional vector indicating in which directions it is stressed
	initialize_3d(size, n_directions, positional_stress); // initialize 3d matrix with 0's
									  //
	// we also need the total stress in each (i,j) to determine which position should undergo swapping
	
	/* print_positional_stress(size, n_directions, positional_stress); */ 
	
	vector<vector<double>> stress;	
	initialize_2d(size, stress); // set elements to 0.0
								 //
	/* print_totalStress(stress); */ 
	
	calculate_stress(src, tar, positional_stress, stress); // stress will be a matrix: each position will carry a stress value
	swap(src, positional_stress, stress, cv);
}



int main(){

	//assumption: 2d matrices are square

	int seed = 10;
	int n_directions = 8;

	try {

		vector<vector<int>> target; // lets first get a matrix which we can index easily, then populate it with whatever elements we need

		// warning: make sure that the same elements exist in target and source. Best thing would be to scramble the target matrix in different ways
		

		//assuming a matrix of size 3x3 and n_kinds = 2	
		//set elements of the target;
		target.push_back(vector<int> {1, 0, 0});
		target.push_back(vector<int> {0, 1, 0});
		target.push_back(vector<int> {0, 0, 1});

		srand(seed);

		vector<vector<int>> src = get_individual(target);
		print_individual(src);
		print_individual(target);

		/* cout<<"fitness: "<<fitness(src, target)<<"\n"; */


		apply_competency(src, target, 3, n_directions);


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
