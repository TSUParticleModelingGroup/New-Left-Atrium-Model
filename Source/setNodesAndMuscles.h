/*
 This file contains all the functions that read in the nodes and muscles, links them together, 
 sets up the node and muscle attributes, and assigns them their values in our units. 
 Additionally it sets any remaining run parameters to get started in the setRemainingParameters() 
 function.
 
 The functions are listed below in the order they appear.
 
 void setNodesFromBlenderFile();
 void checkNodes();
 void setMusclesFromBlenderFile();
 void linkNodesToMuscles();
 double croppedRandomNumber(double, double, double);
 void setRemainingNodeAndMuscleAttributes();
 void getNodesandMusclesFromPreviousRun(); 				(Previous not Previuos-Kyla )
 void setRemainingParameters();
 void hardCodedAblations();
 void hardCodedPeriodicEctopicEvents();
 void hardCodedIndividualMuscleAttributes();
 void checkMuscle(int);
*/

/*
 This function 
 1. Opens the node file.
 2. Finds the number of nodes, the pulse node, the up and front nodes.
 3. Allocates memory to hold the nodes on the CPU and the GPU
 4. Sets all the nodes to their default or start values.
 5. Reads and assigns the node positions from the node file.
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
	Node = (nodeAttributesStructure*)malloc(NumberOfNodes*sizeof(nodeAttributesStructure)); //should be attributes not atributes,will need to fix below-Kyla
	cudaMalloc((void**)&NodeGPU, NumberOfNodes*sizeof(nodeAttributesStructure));
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
		
		Node[i].isBeatNode = false; // Setting all nodes to start out as not being a beat node.
		Node[i].beatPeriod = -1.0; // Setting bogus number so it will throw a flag later if something happens later on.
		Node[i].beatTimer = -1.0; // Setting bogus number so it will throw a flag later if something happens later on.
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

/* This function checks to see if two nodes are too close relative to all the other nodes 
   in the simulations. 
   1: This for loop finds all the nearest neighbor distances and then it calculates the average of this value. 
      This get a sense of how close nodes are in general. If you have more nodes they are going to be 
      closer together, this number just gets you a scale to compare to.
   2: This for loop checks to see if two nodes are closer than an cutoffDivider times smaller than the 
      average minimal distance. If it is, the nodes are printed out with their separation and a flag is set.
      Adjust the cutoffDivider for tighter and looser tolerances.
   3: If the flag is set, the simulation is terminated so the user can correct the node file that contains the faulty nodes.
*/
void checkNodes()
{
	float dx, dy, dz, d;
	float averageMinSeparation, minSeparation; //seperation is spelt wrong, should be separation-Kyla --Fixed(Mason)
	bool flag;
	float cutoffDivider = 100.0;
	float cutoff;
	
	// 1: Finding average nearest neighbor distance.
	averageMinSeparation = 0;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		minSeparation = FLOATMAX; // Setting min as a huge value just to get it started.
		for(int j = 0; j < NumberOfNodes; j++)
		{
			if(i != j)
			{
				dx = Node[i].position.x - Node[j].position.x;
				dy = Node[i].position.y - Node[j].position.y;
				dz = Node[i].position.z - Node[j].position.z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < minSeparation) 
				{
					minSeparation = d;
				}
			}
		}
		averageMinSeparation += minSeparation;
	}
	averageMinSeparation = averageMinSeparation/NumberOfNodes;
	
	// 2: Checking to see if nodes are too close together.
	cutoff = averageMinSeparation/cutoffDivider;
	flag = false;
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
					flag = true;
				}
			}
		}
	}
	
	// 3: Terminating the simulation if nodes were flagged.
	if(flag == true)
	{
		printf("\n The average nearest separation for all the nodes is %f.", averageMinSeparation);
		printf("\n The cutoff separation was %f.\n\n", averageMinSeparation/10.0);
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
	Muscle = (muscleAttributesStructure*)malloc(NumberOfMuscles*sizeof(muscleAttributesStructure));
	cudaMalloc((void**)&MuscleGPU, NumberOfMuscles*sizeof(muscleAttributesStructure));
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
		Muscle[i].compressionStopFraction = -1.0; //misspell compresion should be compression- Mason
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
			printf("\n You are trying to connect to a node that is out of bounds.\n");
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
 This function: 
 1: Uses the Box-Muller method to create a standard normal random number from two uniform random numbers.
 2: Sets the standard deviation to what was input.
 3: Checks to see if the random number is between the desired numbers. If not throw it away and choose again.
*/
double croppedRandomNumber(double stddev, double left, double right)
{
	double temp1, temp2;
	double randomNumber;
	bool test = false;
			
	while(test == false)
	{
		// Getting two uniform random numbers in [0,1]
		temp1 = ((double) rand() / (RAND_MAX));
		temp2 = ((double) rand() / (RAND_MAX));
		
		// Using Box-Muller to get a standard normally distributed random number (mean = 0, stddev = 1)
		randomNumber = sqrt(-2.0 * log(temp1))*cos(2.0*PI*temp2);
		
		// Setting its Standard Deviation to the the desired value. 
		randomNumber *= stddev;
		
		// Chopping the random number between left and right.  
		if(randomNumber < left || right < randomNumber) test = false;
		else test = true;
	}
	return(randomNumber);	
}

/*
 In this function, we set the remaining value of the nodes and muscle which were not already set in the setNodesFromBlenderFile(), 
 the setMusclesFromBlenderFile(), and the linkNodesToMuscles() functions.
 1: First,we find the length of each individual muscle and sum these up to find the total length of all muscles that represent
    the left atrium. 
 2: This allows us to find the fraction of a single muscle's length compared to the total muscle lengths. We can now multiply this 
    fraction by the mass of the left atrium to get the mass on an individual muscle. 
 3: Next, we use the muscle mass to find the mass of each node by taking half (each muscle is connected to two nodes) the mass of all 
    muscles connected to it. We can then use the ratio of node masses like we used the ratio of muscle length like we did in 2 to 
    find the area of each node.
 4: Here we set the final muscle attributes using the scaling read in from the simulationSetup file. The scaling is used so the user
    can adjust the standard muscle attributes to perform as desired in their simulation. We also add some small random fluctuations
    to these values so the simulation can have some stochastic behavior. If you do not want any stochastic behavior simply set the 
    standard deviation for each muscle attribute to zero in the simulationsetup file.
    
 Note: Muscles do not have mass in the simulation. All the mass is carried in the nodes. Muscles were given mass here to be able to
 generate the node masses and area. We carry the muscle masses forward in the event that we need to generate a muscle ratio in 
 future update to the program.
    
*/
void setRemainingNodeAndMuscleAttributes()
{	
	double stddev, left, right;
	
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
		stddev = MuscleConductionVelocitySTD;
		left = -MuscleConductionVelocitySTD;
		right = MuscleConductionVelocitySTD;
		Muscle[i].conductionVelocity = BaseMuscleConductionVelocity + croppedRandomNumber(stddev, left, right);
		
		Muscle[i].conductionDuration = Muscle[i].naturalLength/Muscle[i].conductionVelocity;
		
		stddev = MuscleRefractoryPeriodSTD;
		left = -MuscleRefractoryPeriodSTD;
		right = MuscleRefractoryPeriodSTD;	
		Muscle[i].refractoryPeriod = BaseMuscleRefractoryPeriod + croppedRandomNumber(stddev, left, right);
		
		stddev = AbsoluteRefractoryPeriodFractionSTD;
		left = -AbsoluteRefractoryPeriodFractionSTD;
		right = AbsoluteRefractoryPeriodFractionSTD;
		Muscle[i].absoluteRefractoryPeriodFraction = BaseAbsoluteRefractoryPeriodFraction + croppedRandomNumber(stddev, left, right);
		
		stddev = MyocyteForcePerMassSTD;
		left = -MyocyteForcePerMassSTD;
		right = MyocyteForcePerMassSTD;
		Muscle[i].contractionStrength = MyocyteForcePerMassMultiplier*(MyocyteForcePerMass + croppedRandomNumber(stddev, left, right))*Muscle[i].mass;
		
		Muscle[i].relaxedStrength = MuscleRelaxedStrengthFraction*Muscle[i].contractionStrength;
		
		stddev = MuscleCompressionStopFractionSTD;
		left = -MuscleCompressionStopFractionSTD;
		right = MuscleCompressionStopFractionSTD;
		Muscle[i].compressionStopFraction = MuscleCompressionStopFraction + croppedRandomNumber(stddev, left, right);
	}
	
	printf("\n All node and muscle attributes have been set.");
}

/*
 This function loads all the node and muscle attributes from a previous run file that was saved.
*/
void getNodesandMusclesFromPreviousRun()
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
	
	Node = (nodeAttributesStructure*)malloc(NumberOfNodes*sizeof(nodeAttributesStructure));
	cudaMalloc((void**)&NodeGPU, NumberOfNodes*sizeof(nodeAttributesStructure));
	cudaErrorCheck(__FILE__, __LINE__);
	
	Muscle = (muscleAttributesStructure*)malloc(NumberOfMuscles*sizeof(muscleAttributesStructure));
	cudaMalloc((void**)&MuscleGPU, NumberOfMuscles*sizeof(muscleAttributesStructure));
	cudaErrorCheck(__FILE__, __LINE__);
	
	fread(Node, sizeof(nodeAttributesStructure), NumberOfNodes, inFile);
  	fread(Muscle, sizeof(muscleAttributesStructure), NumberOfMuscles, inFile);
	fclose(inFile);
	
	printf("\n Nodes and Muscles have been read in from %s.", fileName);	
}

/*
 This function sets any remaining parameters that are not part of the nodes or muscles structures.
 It also sets or initializes the run parameters for this run.
*/
void setRemainingParameters()
{
	// Adjusting blood pressure from millimeters of Mercury to our units. //presure -> pressure -Mason
	DiastolicPressureLA *= 0.000133322387415*PressureMultiplier; 
	SystolicPressureLA  *= 0.000133322387415*PressureMultiplier;
	
	RefractoryPeriodAdjustmentMultiplier = 1.0;
	MuscleConductionVelocityAdjustmentMultiplier = 1.0;
	
	CenterOfSimulation.x = 0.0;
	CenterOfSimulation.y = 0.0;
	CenterOfSimulation.z = 0.0;
	
	AngleOfSimulation.x = 0.0;
	AngleOfSimulation.y = 1.0;
	AngleOfSimulation.z = 0.0;

	DrawTimer = 0; 
	RunTime = 0.0;
	IsPaused = true;
	
	DrawNodesFlag = 0;
	DrawFrontHalfFlag = 0;
	
	MovieIsOn = false;
	AblateModeIs = false;
	EctopicBeatModeIs = false;
	AdjustMuscleAreaModeIs = false;
	AdjustMuscleLineModeIs = false;
	FindNodeModeIs = false;
	EctopicEventModeIs = false;
	MouseFunctionModeIs = false;
	
	HitMultiplier = 0.03;
	MouseZ = RadiusOfLeftAtrium;
	ScrollSpeedToggle = 1;
	ScrollSpeed = 1.0;
	MouseWheelPos = 0;
	
	RecenterCount = 0;
	RecenterRate = 10;
	setView(6);
	
	ViewFlag = 1;
}

/*
 If you know that you want to ablate a set of nodes before the simulation
 starts you can do it here, or just wait and do it in the running simulation.
 Do not ablate the PulsePointNode node unless you want to have a simulation 
 that just sits there.
 
 An example is given and commented out below to work from.
*/
void hardCodedAblations()
{	
	// To ablate a selected node set your index and uncomment this line.
	
	/*
	int index = ???;
	if(0 < index && index < NumberOfNodes)
	{
		Node[index].isAblated = true;
		Node[index].drawNodeIs = true;
		Node[index].color.x = 1.0;
		Node[index].color.y = 1.0;
		Node[index].color.z = 1.0;
	}
	
	if(index == PulsePointNode) 
	{
		printf("\n\n You have ablated the pulse point node in the hardCodedAblations() function.");
		printf("\n If this is what you wanted to do, it's fine.");
		printf("\n If not, change your selection in the code hardCodedAblations() function.");
		printf("\n");
	}
	*/
}

/*
 If you know that you want to set a node to be a pulse node before the simulation
 starts you can do it here, or just wait and do it in the running simulation.
 Do not set the the PulsePointNode node because it has already been set in the 
 setNodesFromBlenderFile() function
 
 An example is given and commented out below to work from.
*/
void hardCodedPeriodicEctopicEvents()
{	
	/*
	int index = ???;
	if(0 < index && index < NumberOfNodes && index != PulsePointNode)
	{
		Node[index].isBeatNode = true;
		Node[index].beatPeriod = ???;
		Node[index].beatTimer = ???;
		Node[index].drawNodeIs = true;
		Node[index].color.x = 1.0;
		Node[index].color.y = 0.0;
		Node[index].color.z = 1.0;
	}
	*/
}

/*
 If you know that you want to set a muscle's attributes before the simulation
 starts you can do it here, or just wait and do it in the running simulation.
 
 An example is given and commented out below to work from.
*/
void hardCodedIndividualMuscleAttributes()
{
	/*
	int index = 100; // Set index to the muscle number you want.
	Muscle[index].conductionVelocity = BaseMuscleConductionVelocity*(10.0);
	Muscle[index].conductionDuration = Muscle[index].naturalLength/Muscle[index].conductionVelocity;
	Muscle[index].refractoryPeriod = BaseMuscleRefractoryPeriod*(10.0);
	checkMuscle(index);
	*/
}		
		
/*
 This code 
 1: Checks to see if the electrical signal goes through the muscle faster than the refractory period.
    If it does not a muscle could fire itself and the signal would just bounce back and forth in the muscle.
    If this is true we just kill the muscle and move on.
 2: If a muscle's relaxed strength is greater than it contraction strength something must have gotten entered
    wrong in the setup file. Here we kill the muscle and move on but we might need to kill the simulation.
 3: If the muscle can contract past half its natural length or cannot contract down to its natural length
    something is wrong in the setup simulation file. Here we kill the muscle and move on.
 4: If the muscle should be greater than half the refractory period and less than the refractory period. 
    If not something is wrong. Here we kill the muscle and move on.
 5: If the muscle's contraction strength is negative something is wrong. Here we kill the muscle and move on.
    
 We left each if statement as a stand alone unit in case the user wants to perform a different act in a selected
 if statement. We could have set a flag and just killed the the muscle after all checks, but this gives move
 flexibility for future directions. 
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
	} 
	// 2:							
	if(Muscle[muscleId].contractionStrength < Muscle[muscleId].relaxedStrength)
	{
	 	printf("\n\n The relaxed repulsion strength of muscle %d is greater than its contraction strength. Rethink your parameters.", muscleId);
	 	printf("\n Muscle %d will be disabled. \n", muscleId);
	 	Muscle[muscleId].isDisabled = true;
	 	Muscle[muscleId].color.x = DeadColor.x;
		Muscle[muscleId].color.y = DeadColor.y;
		Muscle[muscleId].color.z = DeadColor.z;
		Muscle[muscleId].color.w = 1.0;
	} 
	// 3:
	if(Muscle[muscleId].compressionStopFraction < 0.5 || 1.0 < Muscle[muscleId].compressionStopFraction)
	{
		printf("\n\n The compression Stop Fraction for muscle %d is %f. Rethink your parameters.", muscleId, Muscle[muscleId].compressionStopFraction);
	 	printf("\n Muscle %d will be disabled. \n", muscleId);
	 	Muscle[muscleId].isDisabled = true;
	 	Muscle[muscleId].color.x = DeadColor.x;
		Muscle[muscleId].color.y = DeadColor.y;
		Muscle[muscleId].color.z = DeadColor.z;
		Muscle[muscleId].color.w = 1.0;
	}
	// 4:
	if(Muscle[muscleId].absoluteRefractoryPeriodFraction < 0.5 || 1.0 < Muscle[muscleId].absoluteRefractoryPeriodFraction)
	{
		printf("\n\n The absolute refractory period for muscle %d is %f. Rethink your parameters.", muscleId, Muscle[muscleId].compressionStopFraction);
	 	printf("\n Muscle %d will be disabled. \n", muscleId);
	 	Muscle[muscleId].isDisabled = true;
	 	Muscle[muscleId].color.x = DeadColor.x;
		Muscle[muscleId].color.y = DeadColor.y;
		Muscle[muscleId].color.z = DeadColor.z;
		Muscle[muscleId].color.w = 1.0;
	}
	// 5:
	if(Muscle[muscleId].contractionStrength < 0.0)
	{
		printf("\n\n The contraction strength for muscle %d is %f. Rethink your parameters.", muscleId, Muscle[muscleId].compressionStopFraction);
	 	printf("\n Muscle %d will be disabled. \n", muscleId);
	 	Muscle[muscleId].isDisabled = true;
	 	Muscle[muscleId].color.x = DeadColor.x;
		Muscle[muscleId].color.y = DeadColor.y;
		Muscle[muscleId].color.z = DeadColor.z;
		Muscle[muscleId].color.w = 1.0;
	}
}

