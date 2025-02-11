// nvcc SVT.cu -o svt -lglut -lm -lGLU -lGL

/*
 This file contains all the the main controller functions that setup the simulation, then run and manage the simulation.
 The functions are listed below in the order they appear.
 
 void n_body(float);
 void setupCudaInvironment();
 void readSimulationParameters();
 void setup();
 int main(int, char**);
*/

// Local include files
#include "./header.h"
#include "./setNodesAndMuscles.h"
#include "./hardCodedNodeAndMuscleAttribute.h"
#include "./callBackFunctions.h"
#include "./drawAndTerminalFunctions.h"
#include "./cudaFunctions.h"

void n_body(float dt)
{	
	if(Pause != 1)
	{	
	 	//printf("\n GridNodes = %d %d %d BlockNodes = %d %d %d\n", GridNodes.x, GridNodes.y, GridNodes.z,BlockNodes.x,BlockNodes.y,BlockNodes.z);
		if(ContractionType != 0)
		{
			getForces<<<GridNodes, BlockNodes>>>(MuscleGPU, NodeGPU, dt, NumberOfNodes, CenterOfSimulation, BaseMuscleCompresionStopFraction, RadiusOfAtria, DiastolicPressureLA, SystolicPressureLA, ContractionType);
			cudaErrorCheck(__FILE__, __LINE__);
			cudaDeviceSynchronize();
		}
		updateNodes<<<GridNodes, BlockNodes>>>(NodeGPU, NumberOfNodes, MUSCLES_PER_NODE, MuscleGPU, DragMultiplier, dt, RunTime, ContractionType);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaDeviceSynchronize();
		updateMuscles<<<GridMuscles, BlockMuscles>>>(MuscleGPU, NodeGPU, NumberOfMuscles, NumberOfNodes, dt, ReadyColor, ContractingColor, RestingColor, RelativeColor, RelativeRefractoryPeriodFraction);
		cudaErrorCheck(__FILE__, __LINE__);
		cudaDeviceSynchronize();
		RecenterCount++;
		if(RecenterCount == RecenterRate) 
		{
			float4 centerOfMass;
			centerOfMass.x = 0.0;
			centerOfMass.y = 0.0;
			centerOfMass.z = 0.0;
			centerOfMass.w = 0.0;
			recenter<<<1, BLOCKCENTEROFMASS>>>(NodeGPU, NumberOfNodes, centerOfMass, CenterOfSimulation);
			cudaErrorCheck(__FILE__, __LINE__);
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

void setupCudaInvironment()
{
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
	
	if((BLOCKCENTEROFMASS > 0) && (BLOCKCENTEROFMASS & (BLOCKCENTEROFMASS - 1)) != 0) 
	{
        	printf("\nBLOCKCENTEROFMASS = %d. This is not a power of 2.", BLOCKCENTEROFMASS);
        	printf("\nBLOCKCENTEROFMASS must be a power of 2 for the center of mass reduction to work.");
        	printf("\nFix this number in the header.h file and try again.");
        	printf("\nGood Bye.\n");
        	exit(0);
        }
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
		setNodesFromBlenderFile();
		checkNodes();
		setMusclesFromBlenderFile();
		linkNodesToMuscles();
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
		int linksPerNode;
		fread(&linksPerNode, sizeof(int), 1, inFile);
		printf("\n linksPerNode = %d", linksPerNode);
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
		printf("\n Nodes and Muscles have been read in.");
	}
	else
	{
		printf("\n Bad NodesMusclesFileOrPreviousRunsFile type.");
		printf("\n Good Bye.");
		exit(0);
	}
	
	setupCudaInvironment();
	
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
	EctopicEventOnOff = 0;
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
	
	cudaMemcpy(NodeGPU, Node, NumberOfNodes*sizeof(nodeAtributesStructure), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(MuscleGPU, Muscle, NumberOfMuscles*sizeof(muscleAtributesStructure), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
        
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
