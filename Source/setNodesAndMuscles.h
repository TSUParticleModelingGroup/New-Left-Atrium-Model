/*
 This file contains all the functions that read in the nodes and muscles, links them together, 
 sets up the node and muscle atributes, and asigns them there values in our units.
 
 The functions are listed below in the order they appear.
 
 void setNodesFromBlenderFile();
 void checkNodes();
 void setMusclesFromBlenderFile();
 void linkNodesToMuscles();
 double getLogNormal(float);
 void setRemainingNodeAndMuscleAttributes();
 void getNodesandMusclesFromPreviuosRun();
 void checkMuscle(int);
*/

/*
 This function 
 1. Opens the node file.
 2. Finds the number of nodes, the pulse node, the up and front nodes.
 3. Allocates memory to hold the nodes on the CPU and the GPU
 4. Sets all the nodes to their default or start values.
 5. Reads and asigns the node positions from the node file.
*/
void setNodesFromBlenderFile()
{	
	FILE *inFile;
	float x, y, z;
	int id;
	char fileName[256];
	
	// Generating the name of the file that holds the nodes.
	char directory[] = "./NodesMuscles/";
	strcpy(fileName, "");
	strcat(fileName, directory);
	strcat(fileName, NodesMusclesFileName);
	strcat(fileName, "/Nodes");
	
	// Opening the node file.
	inFile = fopen(fileName,"rb");
	if(inFile == NULL)
	{
		printf("\n Can't open Nodes file.\n");
		exit(0);
	}
	
	// Reading the header information.
	fscanf(inFile, "%d", &NumberOfNodes);
	printf("\n NumberOfNodes = %d", NumberOfNodes);
	fscanf(inFile, "%d", &PulsePointNode);
	printf("\n PulsePointNode = %d", PulsePointNode);
	fscanf(inFile, "%d", &UpNode);
	printf("\n UpNode = %d", UpNode);
	fscanf(inFile, "%d", &FrontNode);
	printf("\n FrontNode = %d", FrontNode);
	
	// Allocating memory for the CPU and GPU nodes. 
	Node = (nodeAtributesStructure*)malloc(NumberOfNodes*sizeof(nodeAtributesStructure));
	cudaMalloc((void**)&NodeGPU, NumberOfNodes*sizeof(nodeAtributesStructure));
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Setting all nodes to zero or their default settings; 
	for(int i = 0; i < NumberOfNodes; i++)
	{
		Node[i].position.x = 0.0;
		Node[i].position.y = 0.0;
		Node[i].position.z = 0.0;
		Node[i].position.w = 0.0;
		
		Node[i].velocity.x = 0.0;
		Node[i].velocity.y = 0.0;
		Node[i].velocity.z = 0.0;
		Node[i].velocity.w = 0.0;
		
		Node[i].force.x = 0.0;
		Node[i].force.y = 0.0;
		Node[i].force.z = 0.0;
		Node[i].force.w = 0.0;
		
		Node[i].mass = 0.0;
		Node[i].area = 0.0;
		
		Node[i].isBeatNode = false; // Setting all nodes to start out as not be a beat node.
		Node[i].beatPeriod = -1.0; // Setting bogus number so it will throw a flag later if something happens latter on.
		Node[i].beatTimer = -1.0; // Setting bogus number so it will throw a flag later if something happens latter on.
		Node[i].isFiring = false; // Setting the node fire button to false so it will not fire as soon as it is turned on.
		Node[i].isAblated = false; // Setting all nodes to not ablated.
		Node[i].drawNodeIs = false; // This flag will allow you to draw certain nodes even when the draw nodes flag is set to off. Set it to off to start with.
		
		// Setting all node colors to not ablated (green)
		Node[i].color.x = 0.0;
		Node[i].color.y = 1.0;
		Node[i].color.z = 0.0;
		Node[i].color.w = 0.0;
		
		for(int j = 0; j < MUSCLES_PER_NODE; j++)
		{
			Node[i].muscle[j] = -1; // -1 sets the muscle to not used.
		}
	}
	
	// Reading in the nodes positions.
	for(int i = 0; i < NumberOfNodes; i++)
	{
		fscanf(inFile, "%d %f %f %f", &id, &x, &y, &z);
		
		Node[id].position.x = x;
		Node[id].position.y = y;
		Node[id].position.z = z;
	}
	
	// This is the pulse node that generates the beat.
	Node[PulsePointNode].isBeatNode = true;
	Node[PulsePointNode].beatPeriod = BeatPeriod;
	Node[PulsePointNode].beatTimer = BeatPeriod; // Set the time to BeatPeriod so it will kickoff a beat as soon as it starts.
	
	fclose(inFile);
	printf("\n Blender generated nodes have been created.");
}

/* This functions checks to see if two nodes are too close relative to all the other nodes 
   in the simulations. 
   1: This for loop finds all the nearest nieghbor distances. Then it takes the average of this value. 
      This get a sense of how close nodes are in general. If you have more nodes they are going to be 
      closer together, this number just gets you a scale to compair to.
   2: This for loop checks to see if two nodes are closer than an cutoffDivider times smaller than the 
      average minimal distance. If it is, the nodes are printed out with thier seperation and a flag is set.
      Adjust the cutoffDivider for tighter and looser tollerances.
   3: If the flag was set the simulation is terminated so the user can correct the node file that contains the faulty nodes.
*/
void checkNodes()
{
	float dx, dy, dz, d;
	float averageMinSeperation, minSeperation;
	int flag;
	float cutoffDivider = 100.0;
	float cutoff;
	
	// 1: Finding average nearest nieghbor distance.
	averageMinSeperation = 0;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		minSeperation = 10000000.0; // Setting min as a huge value just to get it started.
		for(int j = 0; j < NumberOfNodes; j++)
		{
			if(i != j)
			{
				dx = Node[i].position.x - Node[j].position.x;
				dy = Node[i].position.y - Node[j].position.y;
				dz = Node[i].position.z - Node[j].position.z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < minSeperation) 
				{
					minSeperation = d;
				}
			}
		}
		averageMinSeperation += minSeperation;
	}
	averageMinSeperation = averageMinSeperation/NumberOfNodes;
	
	// 2: Checking to see if nodes are too close together.
	cutoff = averageMinSeperation/cutoffDivider;
	flag =0;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		for(int j = 0; j < NumberOfNodes; j++)
		{
			if(i != j)
			{
				dx = Node[i].position.x - Node[j].position.x;
				dy = Node[i].position.y - Node[j].position.y;
				dz = Node[i].position.z - Node[j].position.z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < cutoff)
				{
					printf("\n Nodes %d and %d are too close. Their separation is %f", i, j, d);
					flag = 1;
				}
			}
		}
	}
	
	// 3: Terminating the simulation if nodes were flagged.
	if(flag == 1)
	{
		printf("\n The average nearest seperation for all the nodes is %f.", averageMinSeperation);
		printf("\n The cutoff seperation was %f.\n\n", averageMinSeperation/10.0);
		exit(0);
	}
	printf("\n Nodes have been checked for minimal separation.");
}

/*
 This function 
 1. Opens the muscles file.
 2. Reads the number of muscles.
 3. Allocates memory to hold the muscles on the CPU and the GPU
 4. Sets all the muscles to their default or start values.
 5. Reads and connects the muscle to the two nodes it is connected to.
*/
void setMusclesFromBlenderFile()
{	
	FILE *inFile;
	int id, idNode1, idNode2;
	char fileName[256];
    
	// Generating the name of the file that holds the muscles.
	char directory[] = "./NodesMuscles/";
	strcpy(fileName, "");
	strcat(fileName, directory);
	strcat(fileName, NodesMusclesFileName);
	strcat(fileName, "/Muscles");
	
	// Opening the muscle file.
	inFile = fopen(fileName,"rb");
	if (inFile == NULL)
	{
		printf("\n Can't open Muscles file.\n");
		exit(0);
	}
	
	fscanf(inFile, "%d", &NumberOfMuscles);
	printf("\n NumberOfMuscles = %d", NumberOfMuscles);
	
	// Allocating memory for the CPU and GPU muscles. 
	Muscle = (muscleAtributesStructure*)malloc(NumberOfMuscles*sizeof(muscleAtributesStructure));
	cudaMalloc((void**)&MuscleGPU, NumberOfMuscles*sizeof(muscleAtributesStructure));
	cudaErrorCheck(__FILE__, __LINE__);

	// Setting all muscles to their default settings; 
	for(int i = 0; i < NumberOfMuscles; i++)
	{
		Muscle[i].nodeA = -1;
		Muscle[i].nodeB = -1;
		Muscle[i].apNode = -1;
		Muscle[i].isOn = false;
		Muscle[i].isDisabled = false;
		Muscle[i].timer = -1.0;
		Muscle[i].mass = -1.0;
		Muscle[i].naturalLength = -1.0;
		Muscle[i].relaxedStrength = -1.0;
		Muscle[i].compresionStopFraction = -1.0;
		Muscle[i].conductionVelocity = -1.0;
		Muscle[i].conductionDuration = -1.0;
		Muscle[i].refractoryPeriod = -1.0;
		Muscle[i].absoluteRefractoryPeriodFraction = -1.0;
		Muscle[i].contractionStrength = -1.0;
		
		// Setting all muscle colors to ready (red)
		Muscle[i].color.x = 1.0;
		Muscle[i].color.y = 0.0;
		Muscle[i].color.z = 0.0;
		Muscle[i].color.w = 0.0;
	}
	
	// Reading in from the blender file what two nodes the muscle connects.
	for(int i = 0; i < NumberOfMuscles; i++)
	{
		fscanf(inFile, "%d", &id);
		fscanf(inFile, "%d", &idNode1);
		fscanf(inFile, "%d", &idNode2);
		
		if(NumberOfMuscles <= id)
		{
			printf("\n You are trying to create a muscle that is out of bounds.\n");
			exit(0);
		}
		if(NumberOfNodes <= idNode1 || NumberOfNodes <= idNode2)
		{
			printf("\n You are trying to conect to a node that is out of bounds.\n");
			exit(0);
		}
		Muscle[id].nodeA = idNode1;
		Muscle[id].nodeB = idNode2;
	}
	
	fclose(inFile);
	printf("\n Blender generated muscles have been created.");
}

/*
 This function loads each node structure with all the muscles it is connected to.
*/
void linkNodesToMuscles()
{	
	int k;
	// Each node will have a list of muscles they are attached to.
	for(int i = 0; i < NumberOfNodes; i++)
	{
		k = 0;
		for(int j = 0; j < NumberOfMuscles; j++)
		{
			if(Muscle[j].nodeA == i || Muscle[j].nodeB == i) // Checking to see if either end of the muscle is attached to node i.
			{
				if(MUSCLES_PER_NODE < k) // Making sure we do not go out of bounds.
				{
					printf("\n Number of muscles connected to node %d is larger than the allowed number of", i);
					printf("\n muscles connected to a single node.");
					printf("\n If this is not a mistake increase MUSCLES_PER_NODE in the header.h file.");
					exit(0);
				}
				Node[i].muscle[k] = j;
				k++;
			}
		}
	}
	printf("\n Nodes have been linked to muscles");
}

/*
 This function uses the Box-Muller meathod to create a log normal random number from two
 uniform random numbers.
*/
double getLogNormal(float standardDeviation)
{
	time_t t;
	
	// Seading the random number generater.
	srand((unsigned) time(&t));
	double temp1, temp2;
	double randomNumber;
	int test;
	
	/*
	// Using Box-Muller to get a standard normally distributed random numbers 
				// from two uniformlly distributed random numbers.
				randomNumber = cos(2.0*PI*temp2)*sqrt(-2 * log(temp1));
				
				// Log normal
				if (DustDistributionType == 0)
				{
					randomNumber = exp(randomNumber);
					diameter = BaseDustDiameter + DustDiameterStandardDeviation*randomNumber;
					test = 1;
				}
	*/			
				
				
	
	// Getting two uniform random numbers in [0,1]
	temp1 = ((double) rand() / (RAND_MAX));
	temp2 = ((double) rand() / (RAND_MAX));
	test = 0;
	while(test == 0)
	{
		// Getting ride of the end points so now random number is in (0,1)
		if(temp1 == 0 || temp1 == 1 || temp2 == 0 || temp2 == 1) 
		{
			temp1 = ((double) rand() / (RAND_MAX));
			temp2 = ((double) rand() / (RAND_MAX));
			test = 0;
		}
		else
		{
			// Using Box-Muller to get a standard normally distributed random numbers 
			// from two uniformlly distributed random numbers.
			randomNumber = cos(2.0*PI*temp2)*sqrt(-2 * log(temp1));
			// Creating a log-normal distrobution from the normal randon number.
			randomNumber = standardDeviation*exp(randomNumber);
			
			test = 1;
		}

	}
	return(randomNumber);	
}

/*
 In this function we set the remaining value of the nodes and muscle which were not already set in the setNodesFromBlenderFile(), 
 the setMusclesFromBlenderFile(), and the linkNodesToMuscles() functions.
 1: First we finding the length of each individual muscle and sum this up to find the total length of all muscles that represent
    the left atrium. 
 2: This allows us to find the fration of a single muscle's length compaired to the total muscle lengths. We can now multiply this 
    fraction by the mass of the left atrium to get the mass on an indivdual muscle. 
 3: Now we use the muscle mass to find the mass of each node by taking half (each muscle is connected to two nodes) the mass of all 
    muscles connected to it. We can then use the ratio of node masses like we used the ratio of muscle length like we did in 2 to 
    find the area of each node.
 4: Here we set the final muscle atributes using the scaling read in from the simulationSetup file. The scaling is use so the user
    can adjust the standard muscle atributes to preform as desired in their simulation. We also add some small random fluctuations
    to these values so the simulation can have some stocastic behavior. If you do not want any stocastic behavior simply set the 
    standard deviation for each muscle attribute to zero in the simulationsetup file.
    
 Note: Muscle do not have mass in the simulation. All the mass is carried in the nodes. Muscles were given mass here to be able to
 generate the node masses and area. We carry the muscle masses forward in the even that we need to generate a muscle ratio in 
 future update to the program.
    
*/
void setRemainingNodeAndMuscleAttributes()
{	
//MuscleConductionVelocitySTD
//AbsoluteRefractoryPeriodFractionSTD 
//MuscleRefractoryPeriodSTD
//MuscleCompresionStopFractionSTD
//MyocyteForcePerMassSTD

	time_t t;
	// Seading the random number generater.
	srand((unsigned) time(&t));
	
	// 1:
	double dx, dy, dz, d;
	double totalLengthOfAllMuscles = 0.0;
	for(int i = 0; i < NumberOfMuscles; i++)
	{	
		dx = Node[Muscle[i].nodeA].position.x - Node[Muscle[i].nodeB].position.x;
		dy = Node[Muscle[i].nodeA].position.y - Node[Muscle[i].nodeB].position.y;
		dz = Node[Muscle[i].nodeA].position.z - Node[Muscle[i].nodeB].position.z;
		d = sqrt(dx*dx + dy*dy + dz*dz);
		Muscle[i].naturalLength = d;
		totalLengthOfAllMuscles += d;
	}
		
	// 2:
	for(int i = 0; i < NumberOfMuscles; i++)
	{
		Muscle[i].mass = MassOfLeftAtrium*(Muscle[i].naturalLength/totalLengthOfAllMuscles);
	}

	// 3:
	float surfaceAreaOfLeftAtrium = 4.0*PI*RadiusOfLeftAtrium*RadiusOfLeftAtrium;
	float ConnectedMuscleMass;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		ConnectedMuscleMass = 0.0;
		for(int j = 0; j < MUSCLES_PER_NODE; j++)
		{
			if(Node[i].muscle[j] != -1)
			{
				ConnectedMuscleMass += Muscle[Node[i].muscle[j]].mass;
			}
		}
		Node[i].mass = ConnectedMuscleMass/2.0;
		Node[i].area = surfaceAreaOfLeftAtrium*(Node[i].mass/MassOfLeftAtrium);
	}
	
	// 4:
	for(int i = 0; i < NumberOfMuscles; i++)
	{	
		Muscle[i].conductionVelocity = BaseMuscleConductionVelocity + getLogNormal(MuscleConductionVelocitySTD);
		Muscle[i].conductionDuration = Muscle[i].naturalLength/Muscle[i].conductionVelocity;	
		Muscle[i].refractoryPeriod = BaseMuscleRefractoryPeriod + getLogNormal(MuscleRefractoryPeriodSTD);
		Muscle[i].absoluteRefractoryPeriodFraction = AbsoluteRefractoryPeriodFraction + getLogNormal(AbsoluteRefractoryPeriodFractionSTD);
		
		Muscle[i].contractionStrength = MyocyteForcePerMassMultiplier*(MyocyteForcePerMass + getLogNormal(MyocyteForcePerMassSTD))*Muscle[i].mass;
		Muscle[i].relaxedStrength = MuscleRelaxedStrengthFraction*Muscle[i].contractionStrength;
		Muscle[i].compresionStopFraction = MuscleCompresionStopFraction + getLogNormal(MuscleCompresionStopFractionSTD);
	}
	
	// Adjusting blood presure from millimeters of Mercury to our units.
	DiastolicPressureLA *= 0.000133322387415*PressureMultiplier; 
	SystolicPressureLA  *= 0.000133322387415*PressureMultiplier;
	
	printf("\n All node and muscle attributes have been set.");
}

/*
 This function load all the node and muscle attributes from a previuos run file that was saved.
*/
void getNodesandMusclesFromPreviuosRun()
{
	FILE *inFile;
	char fileName[256];
	
	strcpy(fileName, "");
	strcat(fileName,"./PreviousRunsFile/");
	strcat(fileName,PreviousRunFileName);
	strcat(fileName,"/run");

	inFile = fopen(fileName,"rb");
	if(inFile == NULL)
	{
		printf(" Can't open %s file.\n", fileName);
		exit(0);
	}
	
	fread(&PulsePointNode, sizeof(int), 1, inFile);
	fread(&UpNode, sizeof(int), 1, inFile);
	fread(&FrontNode, sizeof(int), 1, inFile);
	fread(&NumberOfNodes, sizeof(int), 1, inFile);
	fread(&NumberOfMuscles, sizeof(int), 1, inFile);
	int linksPerNode;
	fread(&linksPerNode, sizeof(int), 1, inFile);
	if(linksPerNode != MUSCLES_PER_NODE)
	{
		printf("\n The number Of muscle per node do not match");
		printf("\n You will have to set the #define MUSCLES_PER_NODE");
		printf("\n to %d in header.h then recompile the code", linksPerNode);
		printf("\n Good Bye\n");
		exit(0);
	}
	
	Node = (nodeAtributesStructure*)malloc(NumberOfNodes*sizeof(nodeAtributesStructure));
	cudaMalloc((void**)&NodeGPU, NumberOfNodes*sizeof(nodeAtributesStructure));
	cudaErrorCheck(__FILE__, __LINE__);
	Muscle = (muscleAtributesStructure*)malloc(NumberOfMuscles*sizeof(muscleAtributesStructure));
	cudaMalloc((void**)&MuscleGPU, NumberOfMuscles*sizeof(muscleAtributesStructure));
	cudaErrorCheck(__FILE__, __LINE__);
	fread(Node, sizeof(nodeAtributesStructure), NumberOfNodes, inFile);
  	fread(Muscle, sizeof(muscleAtributesStructure), NumberOfMuscles, inFile);
	fclose(inFile);
	printf("\n Nodes and Muscles have been read in from %s.", fileName);	
}		
		
/*
 This code 
 1: Checks to see if the electrical signal goes through the muscle faster than the refractory period.
    If it does not a muscle could fire itself and the signal would just bounce back and forth in the muscle.
    If this is true we just kill the muscle and move on.
 2: If a muscles relaxed strength is greater than it contraction strength something must have gotten entered
    wrong in the setup file. Here we kill the muscle and move on but we might should kill the simulation.
 3: If the muscle can contract past half its natural length or cannot contract down to its natural length
    something is wrong in the setup simulation file. Here we kill the muscle and move on.
 We left each if statement as a stand alone unit incase the user wants to perform a different act in a selected
 if statement. We could have set a flag and just killed the the muscle after all checks, but this gives move
 flexability for future directions. 
*/
void checkMuscle(int muscleId)
{
	// 1:
	if(Muscle[muscleId].refractoryPeriod < Muscle[muscleId].conductionDuration)
	{
	 	printf("\n\n Refractory period is shorter than the contraction duration in muscle number %d", muscleId);
	 	printf("\n Muscle %d will be disabled. \n", muscleId);
	 	Muscle[muscleId].isDisabled = true;
	 	Muscle[muscleId].color.x = DeadColor.x;
		Muscle[muscleId].color.y = DeadColor.y;
		Muscle[muscleId].color.z = DeadColor.z;
		Muscle[muscleId].color.w = 1.0;
		sleep(2);
	} 
	// 2:							
	if(Muscle[muscleId].contractionStrength < Muscle[muscleId].relaxedStrength)
	{
	 	printf("\n\n The relaxed repultion strenrth of muscle %d is greater than its contraction strength. Rethink your parameters.", muscleId);
	 	printf("\n Muscle %d will be disabled. \n", muscleId);
	 	Muscle[muscleId].isDisabled = true;
	 	Muscle[muscleId].color.x = DeadColor.x;
		Muscle[muscleId].color.y = DeadColor.y;
		Muscle[muscleId].color.z = DeadColor.z;
		Muscle[muscleId].color.w = 1.0;
		sleep(2);
	} 
	// 3:
	if(Muscle[muscleId].compresionStopFraction < 0.5 || 1.0 < Muscle[muscleId].compresionStopFraction)
	{
		printf("\n\n The compression Stop Fraction for muscle %d is %f. Rethink your parameters.", muscleId, Muscle[muscleId].compresionStopFraction);
	 	printf("\n Muscle %d will be disabled. \n", muscleId);
	 	Muscle[muscleId].isDisabled = true;
	 	Muscle[muscleId].color.x = DeadColor.x;
		Muscle[muscleId].color.y = DeadColor.y;
		Muscle[muscleId].color.z = DeadColor.z;
		Muscle[muscleId].color.w = 1.0;
		sleep(2);
	}
}

