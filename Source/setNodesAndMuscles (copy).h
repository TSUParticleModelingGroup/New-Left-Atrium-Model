/*
 This file contains all the functions that read in the nodes and muscle, links them together, 
 sets up the node and muscle atributes, and asigns them there values in our units.
 The functions are listed below in the order they appear.
 
 void setNodesAndEdgesFromBlenderFile();
 void checkNodes();
 int findNumberOfMuscles();
 void linkMusclesToNodes();
 void linkNodesToMuscles();
 double getLogNormal();
 void setMuscleAttributesAndNodeMasses();
*/

void setNodesAndEdgesFromBlenderFile()
{	
	FILE *inFile;

	float x, y, z;
	int id, idNode1, idNode2;
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
	
	// Allocating memory for the CPU nodes and connections. 
	// Connections will be used in the functions of this file to setup the nodes and muscles then not used agian.
	Node = (nodeAtributesStructure*)malloc(NumberOfNodes*sizeof(nodeAtributesStructure));
	ConnectingNodes = (int*)malloc(NumberOfNodes*MUSCLES_PER_NODE*sizeof(int));
	
	// Setting all nodes to zero or their default settings; 
	for(int i = 0; i < NumberOfNodes; i++)
	{
		Node[id].position.x = 0.0;
		Node[id].position.y = 0.0;
		Node[id].position.z = 0.0;
		Node[id].position.w = 0.0;
		
		Node[i].velocity.y = 0.0;
		Node[i].velocity.x = 0.0;
		Node[i].velocity.z = 0.0;
		Node[i].velocity.w = 0.0;
		
		Node[i].force.y = 0.0;
		Node[i].force.x = 0.0;
		Node[i].force.z = 0.0;
		Node[i].force.w = 0.0;
		
		Node[i].mass = 0.0;
		Node[i].area = 0.0;
		
		Node[i].beatNode = false; // Setting all nodes to start out as not be a beat node.
		Node[i].beatPeriod = -1.0; // Setting bogus number so it will throw a flag later if something happens latter on.
		Node[i].beatTimer = -1.0; // Setting bogus number so it will throw a flag later if something happens latter on.
		Node[i].fire = false; // Setting the node fire button to false so it will not fire as soon as it is turned on.
		Node[i].ablated = false; // Setting all nodes to not ablated.
		Node[i].drawNode = false; // This flag will allow you to draw certain nodes even when the draw nodes flag is set to off. Set it to off to start with.
		
		// Setting all node colors to not ablated (green)
		Node[i].color.y = 0.0;
		Node[i].color.x = 1.0;
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

	fclose(inFile);
    
	// Setting the nodes to -1 so you can tell the nodes that where not used.
	for(int i = 0; i < NumberOfNodes; i++)
	{
		for(int j = 0; j < MUSCLES_PER_NODE; j++)
		{
			ConnectingNodes[i*MUSCLES_PER_NODE + j] = -1;
		}	
	}
	
	// Generating the name of the file that holds the muscles.
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
	
	int used, linkId;
	fscanf(inFile, "%d", &NumberOfMuscles);
	printf("\n NumberOfMuscles = %d", NumberOfMuscles);
	for(int i = 0; i < NumberOfMuscles; i++)
	{
		fscanf(inFile, "%d", &id);
		fscanf(inFile, "%d", &idNode1);
		fscanf(inFile, "%d", &idNode2);
		
		used = 0;
		linkId = 0;
		while(used != 1 && linkId < MUSCLES_PER_NODE)
		{
			if(ConnectingNodes[idNode1*MUSCLES_PER_NODE + linkId] == -1) 
			{
				ConnectingNodes[idNode1*MUSCLES_PER_NODE + linkId] = idNode2;
				used = 1;
			}
			else
			{
				linkId++;
			}
		}
		
		used = 0;
		linkId = 0;
		while(used != 1 && linkId < MUSCLES_PER_NODE)
		{
			if(ConnectingNodes[idNode2*MUSCLES_PER_NODE + linkId] == -1) 
			{
				ConnectingNodes[idNode2*MUSCLES_PER_NODE + linkId] = idNode1;
				used = 1;
			}
			else
			{
				linkId++;
			}
		}
	}
	
	fclose(inFile);
	
	strcpy(fileName, "");
	strcat(fileName,NodesMusclesFileName);
	strcat(fileName,"/Nodes");
	checkNodes();
	
	printf("\n Blender generated nodes and links have been created.");
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

int findNumberOfMuscles()
{
	int count = 0;
	
	for(int i = 0; i < NumberOfNodes; i++)
	{
		for(int j = 0; j < MUSCLES_PER_NODE; j++)
		{
			if(ConnectingNodes[i*MUSCLES_PER_NODE + j] != -1 && ConnectingNodes[i*MUSCLES_PER_NODE + j] > i)
			{
				count++;
			}
		}
	}
	
	return(count);
}

// This code numbers the muscles and connects each end of a muscle to a node.
void linkMusclesToNodes()
{
	int nodeNumberToLinkTo;
	//Setting the ends of the muscles to nodes
	int index = 0;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		for(int j = 0; j < MUSCLES_PER_NODE; j++)
		{
			if(NumberOfNodes*MUSCLES_PER_NODE <= (i*MUSCLES_PER_NODE + j))
			{
				printf("\n TSU Error: number of ConnectingNodes is out of bounds\n");
				exit(0);
			}
			
			nodeNumberToLinkTo = ConnectingNodes[i*MUSCLES_PER_NODE + j];
			
			if(nodeNumberToLinkTo != -1)
			{
				if(i < nodeNumberToLinkTo)
				{
					if(NumberOfMuscles <= index)
					{
						printf("\n TSU Error: number of muscles is out of bounds index = %d\n", index);
						exit(0);
					} 
					Muscle[index].nodeA = i;
					Muscle[index].nodeB = nodeNumberToLinkTo;
					index++;
				}
			}
		}
	}
	
	// Uncomment this to check to see if the muscles are created correctly.
	/*
	for(int i = 0; i < NumberOfMuscles; i++)
	{
		printf("\n Muscle[%d].nodeA = %d  Muscle[%d].nodeB = %d", i, Muscle[i].nodeA, i, Muscle[i].nodeB);
	}
	*/
	
	printf("\n Muscles have been linked to Nodes");
}

// This code connects the newly numbered muscles to the nodes. The nodes know they are connected but they don't the number of the muscle.
void linkNodesToMuscles()
{	
	int nodeNumber;
	// Each node will have a list of muscles they are attached to.
	for(int i = 0; i < NumberOfNodes; i++)
	{
		for(int j = 0; j < MUSCLES_PER_NODE; j++)
		{
			if(NumberOfNodes*MUSCLES_PER_NODE <= (i*MUSCLES_PER_NODE + j))
			{
				printf("\n TSU Error: number of ConnectingNodes is out of bounds in function linkNodesToMuscles\n");
				exit(0);
			}
			
			nodeNumber = ConnectingNodes[i*MUSCLES_PER_NODE + j];
			
			if(nodeNumber != -1)
			{
				for(int k = 0; k < NumberOfMuscles; k++)
				{
					if((Muscle[k].nodeA == i && Muscle[k].nodeB == nodeNumber) || (Muscle[k].nodeA == nodeNumber && Muscle[k].nodeB == i))
					{
						Node[i].muscle[j] = k;
					}
				}
			}
			else
			{
				// If the link is not attached to a muscle set it to -1.
				Node[i].muscle[j] = -1;
			}
		}
	}
	
	// Uncomment this to check to see if the nodes are connected to the correct muscles.
	/*
	for(int i = 0; i < NumberOfNodes; i++)
	{
		for(int j = 0; j < MUSCLES_PER_NODE; j++)
		{
			printf("\n Node = %d  link = %d linked Muscle = %d", i, j, Node[i].muscle[j]);
		}	
	}
	*/
	
	printf("\n Nodes have been linked to muscles");
}

double getLogNormal()
{
	//time_t t;
	// Seading the random number generater.
	//srand((unsigned) time(&t));
	double temp1, temp2;
	double randomNumber;
	int test;
	
	// Getting two uniform random numbers in [0,1]
	temp1 = ((double) rand() / (RAND_MAX));
	temp2 = ((double) rand() / (RAND_MAX));
	test = 0;
	while(test ==0)
	{
		// Getting ride of the end points so now random number is in (0,1)
		if(temp1 == 0 || temp1 == 1 || temp2 == 0 || temp2 == 1) 
		{
			test = 0;
		}
		else
		{
			// Using Box-Muller to get a standard normal random number.
			randomNumber = cos(2.0*PI*temp2)*sqrt(-2 * log(temp1));
			// Creating a log-normal distrobution from the normal randon number.
			randomNumber = exp(randomNumber);
			test = 1;
		}

	}
	return(randomNumber);	
}

void setMuscleAttributesAndNodeMasses()
{	
	float dx, dy, dz, d;
	float sum, totalLengthOfAllMuscles;
	float totalSurfaceAreaUsed, totalMassUsed;
	int count;
	float averageRadius, areaSum, areaAdjustment;
	int k;
	int muscleNumber;
	time_t t;
	int muscleTest, muscleTryCount, muscleTryCountMax;
	
	muscleTryCountMax = 100;
	
	// Seading the random number generater.
	srand((unsigned) time(&t));
	
	// Mogy need to work on this ??????
	// Getting some method of asigning an area to a node so we can get a force from the pressure.
	// We are letting the area be the circle made from the average radius out from a the node in question.
	// This will leave some area left out so we will perportionatly distribute this extra area out to the nodes as well.
	// If shape is a circle first we divide by the number of divsions to get the surface area of a great circler.
	// Then scale by the ratio of the circle compared to a great circle.
	// Circles seem to handle less pressure with this sceam so we downed the pressure by 2/3
	// On the atria1 shape we took out the area that the superiorVenaCava and inferiorVenaCava cover. 
	totalSurfaceAreaUsed = 4.0*PI*RadiusOfAtria*RadiusOfAtria;
	
	// Now we are finding the average radius from a node out to all it's nieghbor nodes.
	// Then setting its area to a circle of half this radius.
	areaSum = 0.0;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		averageRadius = 0.0;
		count = 0;
		for(int j = 0; j < MUSCLES_PER_NODE; j++)
		{
			if(NumberOfNodes*MUSCLES_PER_NODE <= (i*MUSCLES_PER_NODE + j))
			{
				printf("\n TSU Error: number of ConnectingNodes is out of bounds in function setMuscleAttributesAndNodeMasses\n");
				exit(0);
			}
			
			muscleNumber = Node[i].muscle[j];
			if(muscleNumber != -1)
			{
				// The muscle is connected to two nodes. One to me and one to you. Need to find out who you are and not connect to myself.
				k = Muscle[muscleNumber].nodeA;
				if(k == i) k = Muscle[muscleNumber].nodeB;
				dx = Node[k].position.x - Node[i].position.x;
				dy = Node[k].position.y - Node[i].position.y;
				dz = Node[k].position.z - Node[i].position.z;
				averageRadius += sqrt(dx*dx + dy*dy + dz*dz);
				count++;
			}
		}
		if(count != 0) 
		{
			averageRadius /= count; // Getting the average radius; 
			averageRadius /= 2.0; // taking half that radius; 
			Node[i].area = PI*averageRadius*averageRadius; 
		}
		else
		{
			Node[i].area = 0.0; 
		}
		areaSum += Node[i].area;
	}
	
	areaAdjustment = totalSurfaceAreaUsed - areaSum;
	if(0.0 < areaAdjustment)
	{
		for(int i = 0; i < NumberOfNodes; i++)
		{
			if(areaSum < 0.00001)
			{
				printf("\n TSU Error: areaSum is too small (%f)\n", areaSum);
				exit(0);
			}
			else 
			{
				Node[i].area += areaAdjustment*Node[i].area/areaSum;
			}
		}
	}
	
	// Need to work on this Mogy ?????????
	// Setting the total mass used. If it is a sphere it is just the mass of the atria.
	// If shape is a circle first we divide by the number of divsions to get the mass a great circler.
	// Then scale by the ratio of the circle compared to a great circle.
	totalMassUsed = MassOfAtria; 
	// Taking out the mass of the two vena cava holes. It should be the same ration as the ratio of the surface areas.
	totalMassUsed *= totalSurfaceAreaUsed/(4.0*PI*RadiusOfAtria*RadiusOfAtria);
	
	//Finding the length of each muscle and the total length of all muscles.
	totalLengthOfAllMuscles = 0.0;
	for(int i = 0; i < NumberOfMuscles; i++)
	{	
		dx = Node[Muscle[i].nodeA].position.x - Node[Muscle[i].nodeB].position.x;
		dy = Node[Muscle[i].nodeA].position.y - Node[Muscle[i].nodeB].position.y;
		dz = Node[Muscle[i].nodeA].position.z - Node[Muscle[i].nodeB].position.z;
		d = sqrt(dx*dx + dy*dy + dz*dz);
		Muscle[i].naturalLength = d;
		totalLengthOfAllMuscles += d;
	}
	
	// Setting the mass of all muscles.
	for(int i = 0; i < NumberOfMuscles; i++)
	{	
		Muscle[i].mass = totalMassUsed*(Muscle[i].naturalLength/totalLengthOfAllMuscles);
	}
	
	// Setting muscle timing functions
	for(int i = 0; i < NumberOfMuscles; i++)
	{
		Muscle[i].apNode = -1;
		Muscle[i].on = false;
		Muscle[i].disabled = false;
		Muscle[i].timer = 0.0;
		
		muscleTest = 0;
		muscleTryCount = 0;
		while(muscleTest == 0)
		{
			Muscle[i].conductionVelocity = BaseMuscleConductionVelocity;
			Muscle[i].conductionDuration = Muscle[i].naturalLength/Muscle[i].conductionVelocity;
			
			Muscle[i].contractionDuration = BaseMuscleContractionDuration;
			
			Muscle[i].rechargeDuration = BaseMuscleRechargeDuration;
			
			// If it takes the electrical wave longer to cross the muscle than it does to get ready 
			// to fire a muscle could excite itself.
			if(Muscle[i].conductionDuration < Muscle[i].contractionDuration + Muscle[i].rechargeDuration)
			{
				muscleTest = 1;
			}
			
			muscleTryCount++;
			if(muscleTryCountMax < muscleTryCount)
			{
				printf(" \n You have tried to create muscle %d over %d times. You need to reset your muscle timing settings.\n", i, muscleTryCountMax);
				printf(" Good Bye\n");
				exit(0);
			}
		}
	}
	
	// Setting strength functions.
	for(int i = 0; i < NumberOfMuscles; i++)
	{	
		Muscle[i].contractionStrength = MyocyteForcePerMassMultiplier*MyocyteForcePerMass*Muscle[i].mass;
		
		Muscle[i].relaxedStrength = BaseMuscleRelaxedStrengthFraction*Muscle[i].contractionStrength;
		
		// Making sure the muscle will not contract too much or get longer when it is suppose to shrink.
		muscleTest = 0;
		muscleTryCount = 0;
		while(muscleTest == 0)
		{
			Muscle[i].compresionStopFraction = BaseMuscleCompresionStopFraction;
			
			if(0.5 < Muscle[i].compresionStopFraction && Muscle[i].compresionStopFraction < 1.0)
			{
				muscleTest = 1;
			}
			
			muscleTryCount++;
			if(muscleTryCountMax < muscleTryCount)
			{
				printf(" \n You have tried to create muscle %d over %d times. You need to reset your muscle contraction length settings.\n", i, muscleTryCountMax);
				printf(" Good Bye\n");
				exit(0);
			}
		}
	}
	
	// Setting the display color of the muscle.
	for(int i = 0; i < NumberOfMuscles; i++)
	{	
		Muscle[i].color.x = ReadyColor.x;
		Muscle[i].color.y = ReadyColor.y;
		Muscle[i].color.z = ReadyColor.z;
		Muscle[i].color.w = 0.0;
	}
	
	// Setting the node masses
	for(int i = 0; i < NumberOfNodes; i++)
	{
		sum = 0.0;
		for(int j = 0; j < MUSCLES_PER_NODE; j++)
		{
			if(Node[i].muscle[j] != -1)
			{
				sum += Muscle[Node[i].muscle[j]].mass;
			}
		}
		Node[i].mass = sum/2.0;
	}
	printf("\n Muscle Attributes And Node Masses have been set");
}

