//author: Lakshwin
//date: 24th July 2023 @ 1500
//location: Paris, Cite universite

#include <iostream>
#include <math.h>
#include <fstream>
#include <chrono>
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

class index_position{
	public:
		int i = 0;
		int j = 0;
		double max = -100.0;
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
	cout<<"\n=======\n";
}

double fitness(vector<vector<int>> &source, vector<vector<int>> &target){

	//implements eucledian, l2 loss between src and target matrices of the same size, in l2_loss: lower = better; 
	// returns a value where higher = better, to match the definition of fitness 

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

	return 1 - (error/double(pow(source.size(), 2)));

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
		if (neighbors.src_neighbors[i] >=0 || neighbors.tar_neighbors[i] >= 0) count ++;
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


index_position get_max(vector<vector<double>> &stress){
	// stress is pre-conditioned in apply_competency
	int max_i = 0, max_j = 0;

	double max_val = -1000.0;

	for(int i = 0; i<int(stress.size()); ++i){
		for (int j = 0; j<int(stress.size()); ++j){
			if (stress[i][j] > max_val){
				max_val = stress[i][j];
				max_i = i;
				max_j = j;
			}
		}
	}
	index_position temp;
	temp.i = max_i;
	temp.j = max_j;
	temp.max = max_val;

	return temp;
}

index_position get_idxs_from_directions(index_position current_pos, int swap_direction){
	//src and swap_direction are preconditioned (warning: cannot be called separtely from apply_competency)
	
	index_position temp;
	// based on swap_pos index, choose cooresponding matrix coordinates
	if (swap_direction ==  0){
		//north neighbor
		temp.i = current_pos.i-1;
		temp.j = current_pos.j;	
	}

	else if (swap_direction == 1){
		//south 
		temp.i = current_pos.i + 1;
		temp.j = current_pos.j;
	}

	else if (swap_direction == 2){
		//west
		temp.i = current_pos.i;
		temp.j = current_pos.j-1;
	}

	else if (swap_direction == 3){
		//east
		temp.i = current_pos.i;
		temp.j = current_pos.j +1;
	}

	else if (swap_direction == 4){
		//nw
		temp.i = current_pos.i -1;
		temp.j = current_pos.j -1;
	}

	else if (swap_direction == 5){
		//ne
		temp.i = current_pos.i -1;
		temp.j = current_pos.j +1;
	}

	else if (swap_direction == 6){
		//sw
		temp.i = current_pos.i +1;
		temp.j = current_pos.j -1;
	}

	else if (swap_direction == 7){
		//se
		temp.i = current_pos.i +1;
		temp.j = current_pos.j +1;
	}

	return temp;
}

vector<vector<int>> matrix_element_swap(vector<vector<int>> temp_src, index_position curr_pos ,int swap_pos){ // pass by reference is intentionally disabled 
	if (swap_pos <0) throw runtime_error("swap direction must be positive\n");

	index_position new_pos = get_idxs_from_directions(curr_pos, swap_pos);

	double temp_val = 0.0;
	temp_val = temp_src[curr_pos.i][curr_pos.j];
	temp_src[curr_pos.i][curr_pos.j] = temp_src[new_pos.i][new_pos.j];
	temp_src[new_pos.i][new_pos.j] = temp_val;

	return temp_src;

}

int swap(vector<vector<int>> &src, vector<vector<int>> &tar, vector<vector<vector<double>>> &positional_stress, vector<vector<double>> &stress, int cv, int n_directions){

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
	//  Also, Swaps cannot be with the max stress inducing neighbor, it has to be random so that cyclic behviour does not result (swapping between two positions)
	
	int n_swaps = 0;
	int idx_i = -100, idx_j = -100;

	index_position curr_pos = get_max(stress);

	vector<vector<vector<double>>> temp_ps = positional_stress;
	vector<vector<double>> temp_stress = stress;
	vector<vector<int>> temp_src = src;

	
	while ((n_swaps <cv) && (curr_pos.max != 0.0)){
		// if there are no swaps left then make sure the competency value is not mutated anymore, perhaps include a flag
		//
		
		//get idxs of most stressed cell 
		idx_i = curr_pos.i;
		idx_j = curr_pos.j;

		// get the current stress level
		double current_stress = stress[idx_i][idx_j];
		double new_stress = current_stress; 

		//pick a random neighbor of the cell from its positional stress matrix
		int selected_pos = -100;

		do{

			do{
				selected_pos = roll_dice(n_directions);
			}
			while (positional_stress[idx_i][idx_j][selected_pos] == -1); // make sure you don't pick a neighbor which doesn't exist	
																		 //
			// make sure you swap only if stress at that position decreases
			//
			// first carry out a temporatry swap
			temp_src = matrix_element_swap(src, curr_pos, selected_pos); 
			// recalculate stress 
			calculate_stress (temp_src, tar, temp_ps, temp_stress);
			new_stress = temp_stress[idx_i][idx_j];
		}
		// repeat if new_stress > current_stress
		while (new_stress > current_stress);

		// check once again if new stress is lower than current stress and update relevant matrixes
		if (new_stress <= current_stress){
			src = temp_src;
			positional_stress = temp_ps;
			stress = temp_stress;
			n_swaps += 1;
			current_stress = new_stress;
			curr_pos = get_max(stress); //update our knowledge of max stress

		}	

		else throw runtime_error("the do while loop is not doing its job, check your code\n");

		cout<<"Fitness after: "<<fitness(src, tar)<<"\n";	
	
	}

	return n_swaps;
		
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
	//	
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
	int swaps_executed = swap(src, tar, positional_stress, stress, cv, n_directions);

	/* cout<<"swaps executed: "<<swaps_executed<<" and cv: "<<cv<<"\n"; */
}


vector<vector<vector<int>>> create_population(vector<vector<int>> &tar, int n_individuals){
	// creates a population of matrices
	
	//pre-conditions:
	if (n_individuals<0) throw runtime_error("Number of individuals must be positive\n");
	if (n_individuals ==0) cerr<<"Warning: Number of individuals set to 0\n";

	if (int(tar.size()) ==0) throw runtime_error("Empty Target Matrix\n");

	vector<vector<vector<int>>> pop(n_individuals);
	for (int i = 0; i<n_individuals; ++i){
		pop[i] = get_individual(tar);
	}
	return pop;
}

vector<int> initialize_competency(int n_individuals, int max_competency ,int val){
	//initializes a random/constant competency value to each individual in the population	
	if (n_individuals<=0) throw runtime_error("Number of individuals must be atleast 1\n");
	if (val <-1) throw runtime_error("Illegal value. Use -1 for random initialization, and a +ve constant for constant initializaiton\n");

	vector<int> cvs(n_individuals); //sets n_individuals values to zero

	if (val==-1){

		for (int i = 0; i<int(cvs.size()); ++i){
			cvs[i] = roll_dice(max_competency);
		}

	}

	else if (val >=0){
		for (int i = 0; i<int(cvs.size()); ++i){
			cvs[i] = val;
		}
	}

	return cvs;
}

void update_fitness(vector<int> &prev_fitness, vector<vector<vector<int>>> &population, vector<vector<int>> &target){

	// takes a set of fitnesses and updates to them to that of the current population

	if (int(population.size()) == 0) throw runtime_error("Population must have atleast one element\n");
	if (int(prev_fitness.size()) == 0) throw runtime_error("Fitness vector is empty; it must be pre-initialized\n");
	
	for (int i =0; i<int(population.size()); ++i){
		prev_fitness[i] = fitness(population[i], target);	
	}	
}

string create_file(string f_kind){
 
    const auto now = chrono::system_clock::now();
    const time_t t_c = chrono::system_clock::to_time_t(now);
    string curr_time =  ctime(&t_c);

	curr_time = curr_time.substr(0, int(curr_time.size())-1); // exclude the last character because it's a '?'

	string fname = f_kind + '_' + curr_time;

	ofstream customfile(fname);
	customfile.close();

	return fname;
}

void write_to(string hw_fname, vector<int> &fitness){
	//writes hw fitness values on a single line, 
	//with the line number indicating the generation
	//since multiple runs are used, make sure to divide line numbers by the n_runs and plot accordingly
	
	ifstream file_one(hw_fname);
	if (file_one.fail()) throw runtime_error("File does not exist\n");
	if (int(fitness.size()) ==0) throw runtime_error("Fitness vector cannot be empty\n");

	file_one.close();

	ofstream file_two(hw_fname);

	for(int i = 0; i<int(fitness.size()); ++i){
		file_two << fitness[i];
		file_two <<", ";
	}
	file_two<<"\n";
	file_two.close();
}


vector<vector<vector<int>>> iterative_competency(vector<vector<vector<int>>> &population, vector<vector<int>> &target, vector<int> cvs, int n_directions){

	// applies the apply_competency function to each individual of the population

	if(int(population[0].size()) == 0) throw runtime_error("Population cannot be empty\n");
	if(int(target.size()) == 0 ) throw runtime_error("Target matrix cannot be empty\n");
	if(int(cvs.size()) != int(population.size())) throw runtime_error("Number of competency values must be equal to the population size\n");
	for(int x: cvs){
		if (x <0) throw runtime_error("Comeptency values must be >=0\n");
	}
	if(n_directions != 8) throw runtime_error("In a 2D case, the numeber of neighbors cannot be set to anything other than 8\n");

	// copy genotypes so that you can get phenotypes
	vector<vector<vector<int>>> pop_phenotype = population; // because you need the genotypes as well as the phenotypes

	for (int i = 0; i<int(pop_phenotype.size()); ++i){
		apply_competency(pop_phenotype[i], target, cv, n_directions);
	}	

	return pop_phenotype;

}

void sort_population(vector<vector<vector<int>>> &population, vector<int> &fitness){
	//sorts a population in descending order (based on phenotypic fitness)
	
	if(int(population[0].size()) == 0) throw runtime_error("Population cannot be empty\n");
	if(int(fitness.size()) != int(population.size())) throw runtime_error("Fitness array must be of the same size as that of the population\n")

	// bubble sort in descending order
	vector<vector<int>> temp; // placeholder for swapping

	for (int i = 0; i<int(fitness.size())-1; ++i){
		for (int j = i+1; j<int(fitness.size()); ++j){
			if (fitness[i] < fitness[j]){
				temp = population[i];
				population[i] = population[j];
				population[j] = temp;
			}
		}
	}

}

void pick_topk(vector<vector<vector<int>>> &population, double stringency){
	//selects the topk members of a populaiton by deleting the rest

	if(int(population[0].size()) == 0) throw runtime_error("Population cannot be empty\n");
	if (stringency <=0.0 || stringency >1.0) throw runtime_error("Stringency must be between [0, 1.0]\n");

	int start_pos = int(stringency * int(population.size())); 

	int pos_zero = population.begin();

	int terminal_pos = population.end();

	population.erase(pos_zero + start_pos, terminal_pos);

}

void selection(vector<vector<vector<int>>> &population, vector<int> &fitness, double stringency){
	//select the best genotypes based on its phenotypic fitness	
	if(int(population[0].size()) == 0) throw runtime_error("Population cannot be empty\n");
	if(int(fitness.size()) ==0 || int(fitness.size()) != int(population.size())) throw runtime_error("Fitness array must be of the same size as that of the population\n")
	if (stringency <=0.0 || stringency >1.0) throw runtime_error("Stringency must be between [0, 1.0]\n");

	//sort population based on phenotypic fitness. 
		
	sort_population(population, fitness);
	
	//pick the top 10% of the sorted population 
	pick_topk(population, stringency);		

}

bool is_same(vector<vector<int>> v1, vector<vector<int>> v2){
	//checks to see if v1 and v2 have the same elements
	if (int(v1.size()) != int(v2.size())) throw runtime_error("matrices must be of the same size\n");

	int same_counter = 0;
	int size = int(v1.size());

	for (int i = 0; i<size; ++i){
		for (int j = 0; j<size; ++j){
			if (v1[i][j] == v2[i][j]){
				same_counter ++;
			}
		}
	}

	if (same_counter == size*size) return 1;

	else return 0;
}

void mutation_swap(vector<vector<vector<int>>> &population, int idx){
	// swaps values of a matrix until it becomes different 
	
	if(int(population.size())<=0) throw runtime_error("population must have atleast 1 element\n");
	if(idx <0 || idx >int(population.size())) throw runtime_error("idx must be a value of the population\n");

	int temp; //placeholder
		
	int rand_idx_1i = -100, rand_idx_ij = -100; //placeholder inits
	int rand_idx_2i = -100, rand_idx_2j = -100; //placeholder inits
		
	int indv_size = int(population[idx].size()); // matrix size nXn

	vector<vector<int>> old_temp_mat = population[idx]; // chosen matrix to scramble
	vector<vector<int>> new_temp_mat = population[idx]; // chosen matrix to scramble
	
	int temp_val = -1000; //init placeholder (see do-while for usage)

	bool same_flag = 1;

	do{
		rand_idx_1i = roll_dice(indv_size); //pick random x,y coordinate
		rand_idx_1j = roll_dice(indv_size);

		rand_idx_2i = roll_dice(indv_size); //pick another random x,y coordinate
		rand_idx_2j = roll_dice(indv_size);

		temp_val = new_temp_mat[rand_idx_1i][rand_idx_1j]; 
		new_temp_mat[rand_idx_1i][rand_idx_1j] = new_temp_mat[rand_idx_2i][rand_idx_2j]; 
		new_temp_mat[rand_idx_2i][rand_idx_2j] = temp_val;	

		bool same_flag = is_same(new_temp_mat, old_temp_mat); 
	}
	while(same_flag);
	
	//insert new child into the population
	population.push_back(new_temp_mat);
}

void mutate(vector<vector<vector<int>>> &population, int n_individuals, double mutation_prob){
	//mutates a matrix by swapping two of its elements to random positions
	if(int(population.size()) == n_individuals) throw runtime_error("population size must be reduced prior to mutation\n");

	int pop_count = int(population.size());
	int rand_indv = 0; //placeholder

	while(pop_count < n_individuals){

		// swap two random locations based on probability
		if (roll_dice(10)/10.0 <= mutation_prob){
			
			//pick a random individual
			rand_indv = roll_dice(int(pop_count));	

			//scramble that individual in a single position
			mutation_swap(population, rand_indv);
		}
		//post mutation, a new child indvidual exists, so re-calculate size
		pop_count = int(population.size());	
	}
}

void evolve(vector<vector<int>> &target, int n_iterations, int n_individuals, int n_runs, string hw_fname, string comp_fname, int random_init, int max_competency, double stringency, double mutation_prob){

	// pre-conditioning
	if(n_iterations<=0) throw runtime_error("Number of iterations must be >=1\n");
	if(n_individuals<=0) throw runtime_error("Number of individuals must be >0\n");
	if(n_runs <=0) throw runtime_error("Number of runs must be >0\n");
	if (int(hw_fname.size()) ==0 || int(comp_fname.size()) == 0) throw runtime_error("hw / comp filenames must be provided\n");
	if (random_init<-1) throw runtime_error("Random initialization indicator must be -1 or a positive constant\n");
	if (max_competency <=0) throw runtime_error("Maximum competency must be >=1\n");
	if (stringency <=0.0 || stringency >1.0) throw runtime_error("Stringency must be between [0.0, 1.0]\n");
	if (mutation_prob <=0.0 || mutation_prob >1.0) throw runtime_error("Mutation probability must be between [0.0, 1.0]\n");

	//initialize population
	vector<vector<vector<int>>> population = create_population(target, n_individuals);
	vector<vector<vector<int>>> rearranged_pop = population; // initialization for use within the for-loop below

	//initialize competency values
	
	vector<int> competency_values = initialize_competency(n_individuals,  int val = -1); // '-1' stands for random initialization, use a contant for constant initialization.
	
	//initialize fitness vector to zeros
	vector<int> fitness(n_individuals);

	for (int i = 0; i<n_iterations; ++i){
		//calculate initial fitness
		update_fitness(fitness, population, target);

		//write fitness to hardwired fitness filetime_error(o
		write_to(hw_fname, fitness);

		//apply competency based on competency value
		rearranged_pop = iterative_competency(population, target, competency_values, n_directions);
		
		// update phenotypic fitness
		update_fitness(fitness, rearranged_pop, target);

		// write phenotypic fitness to phenotypic fitness file
		write_to(comp_fname, fitness);
		
		// selection based on phenotypic fitness
		selection(population, fitness, stringency) // genotypes are selected based on phenotypic fitness
			
		// mutate population as well as the competency value
		mutate(population, mutation_prob);
	}
}


int main(){

	//assumption: 2d matrices are square

	int seed = 9;
	int n_directions = 8;
	int competency_value = 100;
	int n_individuals = 1000;
	int n_iterations = 1000;
	int n_runs = 4;
	int random_init = -1; // for comepetency initialization 
	int max_competency = 1000;
	double mutation_prob = 0.1;
	double stringency = 0.1;
	string log_dir = "./logs/"; 


	try {

		vector<vector<int>> target; // lets first get a matrix which we can index easily, then populate it with whatever elements we need

		// warning: make sure that the same elements exist in target and source. Best thing would be to scramble the target matrix in different ways
		
		//set elements of the target, the structure of which determines structure of the source. Note n_kinds = 2 for now.
		target.push_back(vector<int> {1, 0, 0, 0, 0, 0, 0, 0, 0, 0});
		target.push_back(vector<int> {0, 1, 0, 0, 0, 0, 0, 0, 0, 0});
		target.push_back(vector<int> {0, 0, 1, 0, 0, 0, 0, 0, 0, 0});

		target.push_back(vector<int> {0, 0, 0, 1, 0, 0, 0, 0, 0, 0});
		target.push_back(vector<int> {0, 0, 0, 0, 1, 0, 0, 0, 0, 0});
		target.push_back(vector<int> {0, 0, 0, 0, 0, 1, 0, 0, 0, 0});

		target.push_back(vector<int> {0, 0, 0, 0, 0, 0, 1, 0, 0, 0});
		target.push_back(vector<int> {0, 0, 0, 0, 0, 0, 0, 1, 0, 0});
		target.push_back(vector<int> {0, 0, 0, 0, 0, 0, 0, 0, 1, 0});

		target.push_back(vector<int> {0, 0, 0, 0, 0, 0, 0, 0, 0, 1});

		string hw_file = create_file(log_dir+"hardwired_fitness");
		string comp_file = create_file(log_dir+"competent_fitness");

		/* vector<vector<int>> src = get_individual(target); */

		srand(seed);

		evolve(target, n_iterations, n_individuals, n_runs, hw_file, comp_file, random_init, max_competency, stringency, mutation_prob);

		
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
