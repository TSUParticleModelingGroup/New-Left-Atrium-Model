#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <GL/glut.h>
#include <GL/freeglut.h>
//#include <GLFW/glfw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <signal.h>
#include <unistd.h>
using namespace std;

// defines for terminal stuff.
#define BOLD_ON  "\e[1m"
#define BOLD_OFF   "\e[m"

// normal defines.
#define PI 3.141592654
#define BLOCKNODES 256
#define BLOCKMUSCLES 256

FILE* MovieFile;
int* Buffer;
int MovieOn;

// CUDA Globals
dim3 BlockNodes, GridNodes;
dim3 BlockMuscles, GridMuscles;

// Timing globals
float Dt;
float PrintRate;
int DrawRate;
int RecenterRate;
int Pause;

// This is the node that the beat iminates from.
int PulsePointNode;

// Nodes that orient the simulation. 
// If UpNode is up and FrontNode is in the front you should be in the standard view.
int UpNode;
int FrontNode;

int AblateOnOff;
int EctopicBeatOnOff;
int AdjustMuscleOnOff;
int FindNodeOnOff;
int EctopicSingleOnOff;
int MouseFunctionOnOff;
int ViewFlag; // 0 orthoganal, 1 fulstum
int MovieFlag; // 0 movie off, 1 movie on
float HitMultiplier;
int ScrollSpeedToggle;
float ScrollSpeed;

int NodesMusclesFileOrPreviousRunsFile;
char NodesMusclesFileName[256];
char PreviousRunFileName[256];
char ViewName[256] = "no view set";
float LineWidth;
int DrawNodesFlag;
float NodeRadiusAdjustment;
int DrawFrontHalfFlag;

float Viscosity;
float MyocyteForcePerMass;
float MyocyteForcePerMassMultiplier;
float DiastolicPressureLA;
float SystolicPressureLA;
float PressureMultiplier;

float BeatPeriod;

float MassOfAtria;
float RadiusOfAtria;

int NumberOfNodes;
int NumberOfMuscles;
int LinksPerNode;
int MaxNumberOfperiodicEctopicEvents;

int ContractionType;
float BaseMuscleRelaxedStrengthFraction;
float BaseMuscleCompresionStopFraction;
float BaseMuscleConductionVelocity;
float BaseMuscleConductionVelocityAdjustmentMultiplier;
float BaseMuscleContractionDuration;
float BaseMuscleContractionDurationAdjustmentMultiplier;
float BaseMuscleRechargeDuration;
float BaseMuscleRechargeDurationAdjustmentMultiplier;
float BaseMuscleContractionStrength;
float RelativeRefractoryPeriodFraction;

float DragMultiplier;

float4 ReadyColor;
float4 ContractingColor;
float4 RestingColor;
float4 RelativeColor;
float4 DeadColor;

float BackGroundRed;
float BackGroundGreen;
float BackGroundBlue;

double MouseX, MouseY, MouseZ;
int MouseWheelPos;

struct muscleAtributesStructure
{
	int nodeA;
	int nodeB;    
	int apNode;
	int onOff;
	int dead;
	float timer;
	float mass;
	float naturalLength;
	float relaxedStrength;
	float compresionStopFraction;
	float conductionVelocity;
	float conductionDuration;
	float contractionDuration;
	float contractionStrength;
	float rechargeDuration;
	float4 color;
};

muscleAtributesStructure *Muscle;
muscleAtributesStructure *MuscleGPU;

struct nodeAtributesStructure
{
	float4 position;
	float4 velocity;
	float4 force;
	float mass;
	float area;
	int ablatedYesNo;
	int drawFlag;
	float4 color;
};

nodeAtributesStructure *Node;
nodeAtributesStructure *NodeGPU;

struct ectopicEventStructure
{
	int node;
	float period;
	float time;
};

ectopicEventStructure *EctopicEvents;
ectopicEventStructure *EctopicEventsGPU;

// This is a list of all the nodes a node is connected to. It is biuld in the initial structure and used to setup the nodes and the muscles
// then it is not used anymore.
int *ConnectingNodes;  

// This is a list of the muscles that each node is connected to.
int *ConnectingMuscles;
int *ConnectingMusclesGPU;

// This will hold the center of mass on the GPU so the center of mass can be adjusted on the GPU. 
// This will keep us from having to copy the nodes down and up to do this on the CPU.
//float4 *CenterOfMassGPU;

float PrintTimer;
int DrawTimer; 
int RecenterCount;
double RunTime;
float4 CenterOfSimulation;
float4 AngleOfSimulation;

// Window globals
static int Window;
//GLFWwindow *Window;
int XWindowSize;
int YWindowSize; 
double Near;
double Far;
double EyeX;
double EyeY;
double EyeZ;
double CenterX;
double CenterY;
double CenterZ;
double UpX;
double UpY;
double UpZ;
	
// Prototyping functions
void allocateMemory(int, int);
int findNumberOfMuscles();
void setNodesAndEdgesLine(float);
void setNodesAndEdgesCircle(float, float); 
void setNodesAndEdgesSphere(int, float);
void setNodesAndEdgesAtria1(int, float, float, float);
void setNodesAndEdgesThickAtria(int, float, float, float);
void linkMusclesToNodes();
void linkNodesToMuscles();
void setMuscleAttributesAndNodeMasses();
void setIndividualMuscleAttributes();
void drawPicture();
void hardCodedAblations();
void hardCodedPeriodicEctopicEvents();
void copyNodesMusclesToGPU();
void copyNodesMusclesFromGPU();
void n_body(float);
void terminalPrint();
void setup();
void orthoganialView();
void fulstrumView();
void KeyPressed(unsigned char, int, int);
void readSimulationParameters();
void errorCheck(const char*);
void checkNodes();
void setView(int);
void adjustView();
// Cuda prototyping
void __global__ getForces(muscleAtributesStructure*, nodeAtributesStructure*, int*, float, int, int, float4, float, float, float, int);
void __global__ updateNodes(nodeAtributesStructure*, int, int, ectopicEventStructure*, int, muscleAtributesStructure*, int*, float, float);
void __global__ updateMuscles(muscleAtributesStructure*, nodeAtributesStructure*, int*, ectopicEventStructure*, int, int, int, int, float);
void __global__ recenter(nodeAtributesStructure*, int, float4, float4);

#include "./setNodesAndMuscles.h"
#include "./callBackFunctions.h"
#include "./screenAndTerminalFunctions.h"
#include "./CUDAFunctions.h"
