// nvcc SVT.cu -o svt -lglut -lm -lGLU -lGL
// -lGL -lm -lX11 -lXrandr -lXi -lXxf86vm -lpthread -ldl

#include "./header.h"

void n_body(float dt)
{	
	if(Pause != 1)
	{	
		if(ContractionType != 0)
		{
			getForces<<<GridNodes, BlockNodes>>>(MuscleGPU, NodeGPU, ConnectingMusclesGPU, dt, NumberOfNodes, LinksPerNode, CenterOfSimulation, BaseMuscleCompresionStopFraction, RadiusOfAtria, DiastolicPressureLA, SystolicPressureLA, ContractionType);
			errorCheck("getForces");
			cudaDeviceSynchronize();
		}
		
		updateNodes<<<GridNodes, BlockNodes>>>(NodeGPU, NumberOfNodes, LinksPerNode, EctopicEventsGPU, MaxNumberOfperiodicEctopicEvents, MuscleGPU, ConnectingMusclesGPU, DragMultiplier, dt, RunTime, ContractionType);
		errorCheck("updateNodes");
		cudaDeviceSynchronize();
		
		updateMuscles<<<GridMuscles, BlockMuscles>>>(MuscleGPU, NodeGPU, ConnectingMusclesGPU, EctopicEventsGPU, NumberOfMuscles, NumberOfNodes, LinksPerNode, MaxNumberOfperiodicEctopicEvents, dt, ReadyColor, ContractingColor, RestingColor, RelativeColor, RelativeRefractoryPeriodFraction);
		errorCheck("updateMuscles");
		cudaDeviceSynchronize();
		
		RecenterCount++;
		if(RecenterCount == RecenterRate) 
		{
			float4 centerOfMass;
			centerOfMass.x = 0.0;
			centerOfMass.y = 0.0;
			centerOfMass.z = 0.0;
			centerOfMass.w = 0.0;
			recenter<<<1, BlockNodes.x>>>(NodeGPU, NumberOfNodes, centerOfMass, CenterOfSimulation);
			errorCheck("recenterGPU");
			RecenterCount = 0;
		}
		
		DrawTimer++;
		if(DrawTimer == DrawRate) 
		{
			copyNodesMusclesFromGPU();
			drawPicture();
			DrawTimer = 0;
		}
		
		PrintTimer += dt;
		if(PrintRate <= PrintTimer) 
		{
			terminalPrint();
			PrintTimer = 0.0;
		}
		
		RunTime += dt; 
	}
	else
	{
		drawPicture();
	}
}

void allocateMemory()
{
	int numberOfMusclesTest;
	setNodesAndEdgesFromBlenderFile();
	
	numberOfMusclesTest = findNumberOfMuscles();
	if(numberOfMusclesTest != NumberOfMuscles)
	{
		printf("\n\nNumber of muscles do not matchup. Something is wrong!\n\n");
		//printf("%d != %d\n\n", numberOfMusclesTest, NumberOfMuscles);
		exit(0);
	}
	Muscle = (muscleAtributesStructure*)malloc(NumberOfMuscles*sizeof(muscleAtributesStructure));
	ConnectingMuscles = (int*)malloc(NumberOfNodes*LinksPerNode*sizeof(int));
	linkMusclesToNodes();
	linkNodesToMuscles();
	
	printf("\n number of nodes = %d", NumberOfNodes);
	printf("\n number of muscles = %d", NumberOfMuscles);
	printf("\n number of links per node = %d", LinksPerNode);
	
	BlockNodes.x = BLOCKNODES;
	BlockNodes.y = 1;
	BlockNodes.z = 1;
	
	GridNodes.x = (NumberOfNodes - 1)/BlockNodes.x + 1;
	GridNodes.y = 1;
	GridNodes.z = 1;
	
	BlockMuscles.x = BLOCKMUSCLES;
	BlockMuscles.y = 1;
	BlockMuscles.z = 1;
	
	GridMuscles.x = (NumberOfMuscles - 1)/BlockMuscles.x + 1;
	GridMuscles.y = 1;
	GridMuscles.z = 1;
	
	//CPU memory is allocated in setNodesAndMuscles.h
	cudaMalloc((void**)&MuscleGPU, NumberOfMuscles*sizeof(muscleAtributesStructure));
	errorCheck("cudaMalloc MuscleGPU");
	//CPU memory is allocated setNodesAndMuscles.h
	cudaMalloc((void**)&NodeGPU, NumberOfNodes*sizeof(nodeAtributesStructure));
	errorCheck("cudaMalloc NodeGPU");
	//CPU memory is allocated setNodesAndMuscles.h
	cudaMalloc((void**)&ConnectingMusclesGPU, NumberOfNodes*LinksPerNode*sizeof(int));
	errorCheck("cudaMalloc ConnectingMusclesGPU");
	// Allocating memory for the ectopic events then setting everything to -1 so we can see that they have not been turned on.
	EctopicEvents = (ectopicEventStructure*)malloc(MaxNumberOfperiodicEctopicEvents*sizeof(ectopicEventStructure));
	cudaMalloc((void**)&EctopicEventsGPU, MaxNumberOfperiodicEctopicEvents*sizeof(ectopicEventStructure));
	errorCheck("cudaMalloc EctopicEventsGPU");
	
	for(int i = 0; i < MaxNumberOfperiodicEctopicEvents; i++)
	{
		EctopicEvents[i].node = -1;
		EctopicEvents[i].period = -1.0;
		EctopicEvents[i].time = -1.0;
	}
	printf("\n Memory has been allocated");
}

void readSimulationParameters()
{
	ifstream data;
	string name;
	
	data.open("./simulationSetup");
	
	if(data.is_open() == 1)
	{
		getline(data,name,'=');
		data >> NodesMusclesFileOrPreviousRunsFile;
		
		getline(data,name,'=');
		data >> NodesMusclesFileName;
		
		getline(data,name,'=');
		data >> PreviousRunFileName;
		
		getline(data,name,'=');
		data >> LineWidth;
		
		getline(data,name,'=');
		data >> NodeRadiusAdjustment;
		
		getline(data,name,'=');
		data >> MyocyteForcePerMass;
		
		getline(data,name,'=');
		data >> MyocyteForcePerMassMultiplier;
		
		getline(data,name,'=');
		data >> DiastolicPressureLA;
		
		getline(data,name,'=');
		data >> SystolicPressureLA;
		
		getline(data,name,'=');
		data >> PressureMultiplier;
		
		getline(data,name,'=');
		data >> MassOfAtria;
		
		getline(data,name,'=');
		data >> RadiusOfAtria;
		
		getline(data,name,'=');
		data >> DragMultiplier;
		
		getline(data,name,'=');
		data >> ContractionType;
		
		getline(data,name,'=');
		data >> BaseMuscleRelaxedStrengthFraction;
		
		getline(data,name,'=');
		data >> BaseMuscleCompresionStopFraction;
		
		getline(data,name,'=');
		data >> BaseMuscleContractionDuration;
		
		getline(data,name,'=');
		data >> BaseMuscleRechargeDuration;
		
		getline(data,name,'=');
		data >> RelativeRefractoryPeriodFraction;
		
		getline(data,name,'=');
		data >> BaseMuscleConductionVelocity;
		
		getline(data,name,'=');
		data >> MaxNumberOfperiodicEctopicEvents;
		
		getline(data,name,'=');
		data >> BeatPeriod;
		
		getline(data,name,'=');
		data >> PrintRate;
		
		getline(data,name,'=');
		data >> DrawRate;
		
		getline(data,name,'=');
		data >> Dt;
		
		getline(data,name,'=');
		data >> ReadyColor.x;
		
		getline(data,name,'=');
		data >> ReadyColor.y;
		
		getline(data,name,'=');
		data >> ReadyColor.z;
		
		getline(data,name,'=');
		data >> ContractingColor.x;
		
		getline(data,name,'=');
		data >> ContractingColor.y;
		
		getline(data,name,'=');
		data >> ContractingColor.z;
		
		getline(data,name,'=');
		data >> RestingColor.x;
		
		getline(data,name,'=');
		data >> RestingColor.y;
		
		getline(data,name,'=');
		data >> RestingColor.z;
		
		getline(data,name,'=');
		data >> RelativeColor.x;
		
		getline(data,name,'=');
		data >> RelativeColor.y;
		
		getline(data,name,'=');
		data >> RelativeColor.z;
		
		getline(data,name,'=');
		data >> DeadColor.x;
		
		getline(data,name,'=');
		data >> DeadColor.y;
		
		getline(data,name,'=');
		data >> DeadColor.z;
		
		getline(data,name,'=');
		data >> BackGroundRed;
		
		getline(data,name,'=');
		data >> BackGroundGreen;
		
		getline(data,name,'=');
		data >> BackGroundBlue;
	}
	else
	{
		printf("\nTSU Error could not open simulationSetup file\n");
		exit(0);
	}
	
	/*
	if(NodesMusclesFileOrPreviousRunsFile == 0)
	{
		printf("\n Object Name = %s", NodesMusclesFileName);
	}
	else if(NodesMusclesFileOrPreviousRunsFile == 1)
	{
		printf("\n Object Name = %s", PreviousRunFileName);
	}
	*/
	
	RecenterRate = 10; 
	
	data.close();
	// Adjusting blood presure from millimeters of Mercury to our units.
	DiastolicPressureLA *= 0.000133322387415*PressureMultiplier; 
	SystolicPressureLA  *= 0.000133322387415*PressureMultiplier;
	
	printf("\n Simulation Parameters have been read in.");
}

void setup()
{	
	readSimulationParameters();
	if(NodesMusclesFileOrPreviousRunsFile == 0)
	{
		allocateMemory();
		setMuscleAttributesAndNodeMasses();
		setIndividualMuscleAttributes();
		hardCodedAblations();
		hardCodedPeriodicEctopicEvents();
	}
	else if(NodesMusclesFileOrPreviousRunsFile == 1)
	{
		FILE *inFile;
		char fileName[256];
		
		strcpy(fileName, "");
		strcat(fileName,"./PreviousRunsFile/");
		strcat(fileName,PreviousRunFileName);
		strcat(fileName,"/run");
		//printf("\n fileName = %s\n", fileName);

		inFile = fopen(fileName,"rb");
		if(inFile == NULL)
		{
			printf(" Can't open %s file.\n", fileName);
			exit(0);
		}
		
		fread(&PulsePointNode, sizeof(int), 1, inFile);
		printf("\n PulsePointNode = %d", PulsePointNode);
		fread(&UpNode, sizeof(int), 1, inFile);
		printf("\n UpNode = %d", UpNode);
		fread(&FrontNode, sizeof(int), 1, inFile);
		printf("\n FrontNode = %d", FrontNode);
		
		fread(&NumberOfNodes, sizeof(int), 1, inFile);
		printf("\n NumberOfNodes = %d", NumberOfNodes);
		fread(&NumberOfMuscles, sizeof(int), 1, inFile);
		printf("\n NumberOfMuscles = %d", NumberOfMuscles);
		fread(&LinksPerNode, sizeof(int), 1, inFile);
		printf("\n LinksPerNode = %d", LinksPerNode);
		fread(&MaxNumberOfperiodicEctopicEvents, sizeof(int), 1, inFile);
		printf("\n MaxNumberOfperiodicEctopicEvents = %d", MaxNumberOfperiodicEctopicEvents);
		
		Node = (nodeAtributesStructure*)malloc(NumberOfNodes*sizeof(nodeAtributesStructure));
		Muscle = (muscleAtributesStructure*)malloc(NumberOfMuscles*sizeof(muscleAtributesStructure));
		ConnectingMuscles = (int*)malloc(NumberOfNodes*LinksPerNode*sizeof(int));
		EctopicEvents = (ectopicEventStructure*)malloc(MaxNumberOfperiodicEctopicEvents*sizeof(ectopicEventStructure));
		
		cudaMalloc((void**)&MuscleGPU, NumberOfMuscles*sizeof(muscleAtributesStructure));
		errorCheck("cudaMalloc MuscleGPU");
		cudaMalloc((void**)&NodeGPU, NumberOfNodes*sizeof(nodeAtributesStructure));
		errorCheck("cudaMalloc NodeGPU");
		cudaMalloc((void**)&ConnectingMusclesGPU, NumberOfNodes*LinksPerNode*sizeof(int));
		errorCheck("cudaMalloc ConnectingMusclesGPU");
		cudaMalloc((void**)&EctopicEventsGPU, MaxNumberOfperiodicEctopicEvents*sizeof(ectopicEventStructure));
		errorCheck("cudaMalloc EctopicEventsGPU");
	
		for(int i = 0; i < MaxNumberOfperiodicEctopicEvents; i++)
		{
			EctopicEvents[i].node = -1;
			EctopicEvents[i].period = -1.0;
			EctopicEvents[i].time = -1.0;
		}
		printf("\n Memory has been allocated");
	
		
		fread(Node, sizeof(nodeAtributesStructure), NumberOfNodes, inFile);
	  	fread(Muscle, sizeof(muscleAtributesStructure), NumberOfMuscles, inFile);
	  	fread(ConnectingMuscles, sizeof(int), NumberOfNodes*LinksPerNode, inFile);
	  	fread(EctopicEvents, sizeof(ectopicEventStructure), MaxNumberOfperiodicEctopicEvents, inFile);
		fclose(inFile);
		printf("\n Nodes and Muscles have been read in.");
		
		BlockNodes.x = BLOCKNODES;
		BlockNodes.y = 1;
		BlockNodes.z = 1;
		
		GridNodes.x = (NumberOfNodes - 1)/BlockNodes.x + 1;
		GridNodes.y = 1;
		GridNodes.z = 1;
		
		BlockMuscles.x = BLOCKMUSCLES;
		BlockMuscles.y = 1;
		BlockMuscles.z = 1;
		
		GridMuscles.x = (NumberOfMuscles - 1)/BlockMuscles.x + 1;
		GridMuscles.y = 1;
		GridMuscles.z = 1;
	}
	else
	{
		printf("\n Bad NodesMusclesFileOrPreviousRunsFile type.");
		printf("\n Good Bye.");
		exit(0);
	}
	
	AngleOfSimulation.x = 0.0;
	AngleOfSimulation.y = 1.0;
	AngleOfSimulation.z = 0.0;
	
	BaseMuscleContractionDurationAdjustmentMultiplier = 1.0;
	BaseMuscleRechargeDurationAdjustmentMultiplier = 1.0;
	BaseMuscleConductionVelocityAdjustmentMultiplier = 1.0;

	DrawTimer = 0; 
	RecenterCount = 0;
	RunTime = 0.0;
	Pause = 1;
	MovieOn = 0;
	DrawNodesFlag = 0;
	DrawFrontHalfFlag = 0;
	
	AblateOnOff = 0;
	EctopicBeatOnOff = 0;
	AdjustMuscleOnOff = 0;
	FindNodeOnOff = 0;
	EctopicSingleOnOff = 0;
	MouseFunctionOnOff = 0;
	ViewFlag = 1;
	MovieFlag = 0;
	HitMultiplier = 0.03;
	MouseZ = RadiusOfAtria;
	ScrollSpeedToggle = 1;
	ScrollSpeed = 1.0;
	MouseWheelPos = 0;
	
	centerObject();
	setView(2);
	
	cudaMemcpy( MuscleGPU, Muscle, NumberOfMuscles*sizeof(muscleAtributesStructure), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy Muscle up");
	cudaMemcpy( NodeGPU, Node, NumberOfNodes*sizeof(nodeAtributesStructure), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy Node up");
	cudaMemcpy( ConnectingMusclesGPU, ConnectingMuscles, NumberOfNodes*LinksPerNode*sizeof(int), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy ConnectingMuscles up");
	cudaMemcpy( EctopicEventsGPU, EctopicEvents, MaxNumberOfperiodicEctopicEvents*sizeof(ectopicEventStructure), cudaMemcpyHostToDevice );
	errorCheck("cudaMemcpy EctopicEvents up");
	
	//printf("\n\n The Particle Modeling Group hopes you enjoy your interactive simulation.");
	//printf("\n The simulation is paused type r to start the simulation and h for a help menu.");
	//printf("\n");
	
	printf("\n");
	terminalPrint();
}

int main(int argc, char** argv)
{
	setup();
	
	XWindowSize = 1000;
	YWindowSize = 1000; 

	// Clip plains
	Near = 0.2;
	Far = 80.0*RadiusOfAtria;

	//Direction here your eye is located location
	EyeX = 0.0*RadiusOfAtria;
	EyeY = 0.0*RadiusOfAtria;
	EyeZ = 2.0*RadiusOfAtria;

	//Where you are looking
	CenterX = 0.0;
	CenterY = 0.0;
	CenterZ = 0.0;

	//Up vector for viewing
	UpX = 0.0;
	UpY = 1.0;
	UpZ = 0.0;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(5,5);
	Window = glutCreateWindow("SVT");
	
	gluLookAt(EyeX, EyeY, EyeZ, CenterX, CenterY, CenterZ, UpX, UpY, UpZ);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	glClearColor(BackGroundRed, BackGroundGreen, BackGroundBlue, 0.0);
	
	//GLfloat light_position[] = {EyeX, EyeY, EyeZ, 0.0};
	GLfloat light_position[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	
	//glutMouseFunc(mouseWheelCallback);
	//glutMouseWheelFunc(mouseWheelCallback);
	//glutMotionFunc(mouseMotionCallback);
    	glutPassiveMotionFunc(mousePassiveMotionCallback);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mymouse);
	glutKeyboardFunc(KeyPressed);
	glutIdleFunc(idle);
	glutSetCursor(GLUT_CURSOR_DESTROY);
	glEnable(GL_DEPTH_TEST);
	
	glutMainLoop();
	return 0;
}
