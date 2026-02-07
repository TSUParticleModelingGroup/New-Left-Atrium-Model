

/*
 This file contains all the functions that read in the nodes and muscles, links them together, 
 sets up the node and muscle attributes, and assigns them their values in our units. 
 Additionally it sets any remaining run parameters to get started in the setRemainingParameters() 
 function.
 
 The functions are listed below in the order they appear.
 
 void setNodesFromBlenderFile();
 void checkNodes();
 void setBachmannBundleFromBlenderFile();
 void setMusclesFromBlenderFile();
 void linkNodesToMuscles();
 double croppedRandomNumber(double, double, double);
 void findRadiusAndMassOfLeftAtrium();
 void setRemainingNodeAndMuscleAttributes();
 void getNodesandMusclesFromPreviousRun();
 void setRemainingParameters();
 void hardCodedAblations();
 void hardCodedPeriodicEctopicEvents();fscanf(inFile, "%d", &NumberOfNodes)
 7. Places the center of the LA at (0,0,0).
 8. Sets the pulse node.
*/
void setNodesFromBlenderFile()
{	
	FILE *inFile;
	float x, y, z;
	int id;
	char fileName[256];
	
	// Generating the name of the file that holds the nodes.
	
	char directory[] = "NodesMuscles/";
	strcpy(fileName, "");
	strcat(fileName, directory);
	strcat(fileName, NodesMusclesFileName);
	strcat(fileName, "/Nodes");
	printf("%s", fileName);
	
	// 1. Opening the node file.
	inFile = fopen(fileName,"rb");
	if(inFile == NULL)
	{
		printf("\n\n Can't open Nodes file %s.", fileName);
		printf("\n The simulation has been terminated.\n\n");
		exit(0);
	}
	
	// 2. Reading the header information.
	fscanf(inFile, "%d", &NumberOfNodes);
	printf("\n NumberOfNodes = %d", NumberOfNodes);
	fscanf(inFile, "%d", &PulsePointNode);
	printf("\n PulsePointNode = %d", PulsePointNode);
	fscanf(inFile, "%d", &UpNode);
	printf("\n UpNode = %d", UpNode);
	fscanf(inFile, "%d", &FrontNode);
	printf("\n FrontNode = %d", FrontNode);
	printf("\n");
	
	// 3. Allocating memory for the CPU and GPU nodes. 
	cudaHostAlloc((void**)&Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaHostAllocDefault); // Making page locked memory on the CPU.
	
	
	// 4. Setting all nodes to zero or their default settings; 
	for(int i = 0; i < NumberOfNodes; i++)
	{
		Node[i].position.x = 0.0;
		Node[i].position.y = 0.0;
		Node[i].position.z = 0.0;
		Node[i].position.w = 0.0;
		
		// Setting all node colors to not ablated (green)
		Node[i].color.x = 0.0;
		Node[i].color.y = 1.0;
		Node[i].color.z = 0.0;
		Node[i].color.w = 0.0;

		Node[i].mass = 0.005266; // We took the average mass per node from the original program. Mass is only used to calculate COM for rotations in this program so any number should work here.
		Node[i].type = 0; // Setting the type to 0 for all nodes. We will use this to flag special nodes, like the pulse node and the bachmann's bundle nodes.	
		for(int j = 0; j < MUSCLES_PER_NODE; j++)
		{
			Node[i].muscle[j] = -1; // -1 sets the muscle to not used.
		}
	}
	
	// 5. Reading in the nodes positions.
	for(int i = 0; i < NumberOfNodes; i++)
	{
		fscanf(inFile, "%d %f %f %f", &id, &x, &y, &z);
		
		Node[id].position.x = x;
		Node[id].position.y = y;
		Node[id].position.z = z;
	}
	
	// 6. Finding center on LA
	float4 centerOfObject;
	centerOfObject.x = 0.0;
	centerOfObject.y = 0.0;
	centerOfObject.z = 0.0;
	centerOfObject.w = (double)NumberOfNodes;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		 centerOfObject.x += Node[i].position.x;
		 centerOfObject.y += Node[i].position.y;
		 centerOfObject.z += Node[i].position.z;
	}
	centerOfObject.x /= centerOfObject.w;
	centerOfObject.y /= centerOfObject.w;
	centerOfObject.z /= centerOfObject.w;
	
	// 7. Centering the LA at (0,0,0)
	for(int i = 0; i < NumberOfNodes; i++)
	{
		Node[i].position.x -= centerOfObject.x;
		Node[i].position.y -= centerOfObject.y;
		Node[i].position.z -= centerOfObject.z;
	}

	//TODO: The .isBeatNode has been changed to a node type, the same as the bachman's bundle nodes.
	// 8. This is the pulse node that generates the beat.
	//Node[PulsePointNode].isBeatNode = true;
	//Node[PulsePointNode].beatPeriod = BeatPeriod;
	// Node[PulsePointNode].beatTimer = BeatPeriod; // Set the time to BeatPeriod so it will kickoff a beat as soon as it starts.
	
	fclose(inFile);
	printf("\n Blender generated nodes have been created.\n");
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
	float averageMinSeparation, minSeparation;
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
		printf("\n\n The average nearest separation for all the nodes is %f.", averageMinSeparation);
		printf("\n The cutoff separation was %f.", cutoff);
		printf("\n The simulation has been terminated.\n\n");
		exit(0);
	}
	
	printf("\n Nodes have been checked for minimal separation.\n");
}

/*
 This function 
 1. Opens the Left Atrial Appendage (LAA) file.
 2. Reads the number of nodes in the LAA.
 3. Allocates memory on both CPU and GPU to hold BB.
 4. Reads the LAA nodes.
 */
void setLeftAtrialAppendageFromBlenderFile()
{	
	FILE *inFile;
	int id;
	char fileName[256];
	
	// Generating the name of the file that holds the nodes.
	char directory[] = "./NodesMuscles/";
	strcpy(fileName, "");
	strcat(fileName, directory);
	strcat(fileName, NodesMusclesFileName);
	strcat(fileName, "/LeftAtrialAppendage");
	
	// Opening the file.
	inFile = fopen(fileName,"rb");
	if(inFile == NULL)
	{
		printf("\n\n Can't open Left Atrial Appendage file.");
		printf("\n The simulation has been terminated.\n\n");
		exit(0);
	}
	
	// Reading the header information.
	fscanf(inFile, "%d", &NumberOfNodesInLeftAtrialAppendage);
	printf("\n NumberOfNodesInBachmannsBundle = %d", NumberOfNodesInLeftAtrialAppendage);
	
	// Allocating memory for the LAA nodes.
	LeftAtrialAppendage = (int*)malloc(NumberOfNodesInLeftAtrialAppendage*sizeof(int));
	
	// If we want to use BB on the GPU use the following and define BachmannsBundleGPU in header.h.
	// Allocating memory for the CPU and GPU LAA nodes. 
	//cudaHostAlloc(&LeftAtrialAppendage, NumberOfNodesInLeftAtrialAppendage*sizeof(int), cudaHostAllocDefault); // Making page locked memory on the CPU.
	////cudaErrorCheck(__FILE__, __LINE__);
	
	//cudaMalloc((void**)&LeftAtrialAppendage, NumberOfNodesInLeftAtrialAppendage*sizeof(int));
	////cudaErrorCheck(__FILE__, __LINE__);
	
	// Reading the nodes that extend from the pulse node to create LeftAtrialAppendage.
	for(int i = 0; i < NumberOfNodesInLeftAtrialAppendage; i++)
	{
		fscanf(inFile, "%d ", &id);
		LeftAtrialAppendage[i] = id;
	}
	
	fclose(inFile);
	printf("\n Left Atrial Appendage nodes have been read in.\n");
}

/*
 This function 
 1. Opens the Bachmann's Bundle (BB) file.
 2. Reads the number of nodes in the BB.
 3. Allocating memory on both CPU and GPU to hold BB.
 4. Reads the BB nodes.
 */
void setBachmannBundleFromBlenderFile()
{	
	FILE *inFile;
	int id;
	char fileName[256];
	
	// Generating the name of the file that holds the nodes.
	char directory[] = "./NodesMuscles/";
	strcpy(fileName, "");
	strcat(fileName, directory);
	strcat(fileName, NodesMusclesFileName);
	strcat(fileName, "/BachmannsBundle");
	
	// Opening the file.
	inFile = fopen(fileName,"rb");
	if(inFile == NULL)
	{
		printf("\n\n Can't open Bachmann's Bundle file.");
		printf("\n The simulation has been terminated.\n\n");
		exit(0);
	}
	
	// Reading the header information.
	fscanf(inFile, "%d", &NumberOfNodesInBachmannsBundle);
	printf("\n NumberOfNodesInBachmannsBundle = %d", NumberOfNodesInBachmannsBundle);
	
	// Allocating memory for the Bachmann's Bundle nodes.
	BachmannsBundle = (int*)malloc(NumberOfNodesInBachmannsBundle*sizeof(int));
	
	// If we want to use BB on the GPU use the following and define BachmannsBundleGPU in header.h.
	// Allocating memory for the CPU and GPU Bachmann's Bundle nodes. 
	//cudaHostAlloc(&BachmannsBundle, NumberOfNodesInBachmannsBundle*sizeof(int), cudaHostAllocDefault); // Making page locked memory on the CPU.
	////cudaErrorCheck(__FILE__, __LINE__);
	
	//cudaMalloc((void**)&BachmannsBundleGPU, NumberOfNodesInBachmannsBundle*sizeof(int));
	////cudaErrorCheck(__FILE__, __LINE__);
	
	// Reading the nodes that extend from the pulse node to create Bachmann's Bundle.
	for(int i = 0; i < NumberOfNodesInBachmannsBundle; i++)
	{
		fscanf(inFile, "%d ", &id);
		BachmannsBundle[i] = id;
	}
	
	fclose(inFile);
	printf("\n Bachmann's Bundle Node have been read in.\n");
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
		printf("\n\n Can't open Muscles file %s.", fileName);
		printf("\n The simulation has been terminated.\n\n");
		exit(0);
	}
	
	fscanf(inFile, "%d", &NumberOfMuscles);
	printf("\n NumberOfMuscles = %d", NumberOfMuscles);
	
	// Allocating memory for the CPU and GPU muscles. 
	//Muscle = (muscleAttributesStructure*)malloc(NumberOfMuscles*sizeof(muscleAttributesStructure));
	cudaHostAlloc(&Muscle, NumberOfMuscles*sizeof(muscleAttributesStructure), cudaHostAllocDefault); // Making page locked memory on the CPU.
	//cudaErrorCheck(__FILE__, __LINE__);

	// Setting all muscles to their default settings; 
	for(int i = 0; i < NumberOfMuscles; i++)
	{
		Muscle[i].nodeA = -1;
		Muscle[i].nodeB = -1;

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
			printf("\n\n You are trying to create a muscle that is out of bounds.");
			printf("\n The simulation has been terminated.\n\n");
			exit(0);
		}
		if(NumberOfNodes <= idNode1 || NumberOfNodes <= idNode2)
		{
			printf("\n\n You are trying to connect to a node that is out of bounds.");
			printf("\n The simulation has been terminated.\n\n");
			exit(0);
		}
		Muscle[id].nodeA = idNode1;
		Muscle[id].nodeB = idNode2;
	}
	
	fclose(inFile);
	printf("\n Blender generated muscles have been created.\n");
}

/*
 This function loads each node structure with all the muscles it is connected to.
*/
void linkNodesToMuscles()
{	
	int k;
	// Each node will have a list of the muscles it is attached to.
	for(int i = 0; i < NumberOfNodes; i++)
	{
		k = 0;
		for(int j = 0; j < NumberOfMuscles; j++)
		{
			if(Muscle[j].nodeA == i || Muscle[j].nodeB == i) // Checking to see if either end of the muscle is attached to node i.
			{
				if(MUSCLES_PER_NODE < k) // Making sure we do not go out of bounds.
				{
					printf("\n\n Number of muscles connected to node %d is larger than the allowed number of", i);
					printf("\n muscles connected to a single node.");
					printf("\n If this is not a mistake increase MUSCLES_PER_NODE in the header.h file.");
					printf("\n The simulation has been terminated.\n\n");
					exit(0);
				}
				Node[i].muscle[k] = j;
				k++;
			}
		}
	}
	printf("\n Nodes have been linked to muscles. \n");
}

/*
 This function sets any remaining parameters that are not part of the nodes or muscles structures.
 It also sets or initializes the run parameters for this run.
*/
void setRemainingParameters()
{	
	// If this is a new run these values are set hre. If it is a previous run these values will aready be read in.
	if (NodesMusclesFileOrPreviousRunsFile == 0) 
	{
		// TODO: See if we can move this to the header as defaults
		CenterOfSimulation.x = 0.0;
		CenterOfSimulation.y = 0.0;
		CenterOfSimulation.z = 0.0;
		CenterOfSimulation.w = 0.0;
		
		AngleOfSimulation.x = 0.0;
		AngleOfSimulation.y = 1.0;
		AngleOfSimulation.z = 0.0;
		AngleOfSimulation.w = 0.0;

		//Simulation.ContractionisOn = false; //This is set in the BasicSimulationSetup file.
		Simulation.ViewFlag = 1;
		Simulation.DrawNodesFlag = 0;
		Simulation.DrawFrontHalfFlag = 0;
		// Simulation.guiCollapsed = false; //This is set in viewDrawAndTerminalFuctions.h/createGUI().
		
		setView(6); //Set deafult view only if not loading from previous run.
	}
	
	HitMultiplier = 0.03;
	MouseZ = RadiusOfLeftAtrium;
	MouseX = 0.0;
	MouseY = 0.0;
	ScrollSpeedToggle = 1;
	ScrollSpeed = 1.0;
	MouseWheelPos = 0;
	RecenterCount = 0;
	RecenterRate = 10;
}


		
		


