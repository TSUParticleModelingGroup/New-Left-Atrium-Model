/*
 This file contains all include files, the #defines, structures and globals used in the simulation.
 All the functions are prototyped in this file as well.
*/

// External include files
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
#include <stdbool.h>

using namespace std;

// Cuda defines
#define BLOCKNODES 256
#define BLOCKMUSCLES 256
#define BLOCKCENTEROFMASS 512
#define FLOATMAX 3.4028235e+38f
#define INTMAX 2147483647

// Defines for terminal print
#define BOLD_ON  "\e[1m"
#define BOLD_OFF   "\e[m"

// Math defines.
#define PI 3.141592654

// Structure defines. 
// This sets how many muscle can be connected to a node.
#define MUSCLES_PER_NODE 20

// Structures
// Everything a node holds. We have 1 on the CPU and 1 on the GPU
struct nodeAttributesStructure
{
	float4 position;
	float4 velocity;
	float4 force;
	float mass;
	float area;
	bool isBeatNode;
	float beatPeriod;
	float beatTimer;
	bool isFiring;
	bool isAblated;
	bool isDrawNode;
	float4 color;
	int muscle[MUSCLES_PER_NODE];
};

// Everything a muscle holds. We have 1 on the CPU and 1 on the GPU
struct muscleAttributesStructure
{
	int nodeA;
	int nodeB;    
	int apNode;
	bool isOn;
	bool isEnabled;
	float timer;
	float mass;
	float naturalLength;
	float relaxedStrength;
	float compressionStopFraction;
	float conductionVelocity;
	float conductionDuration;
	float refractoryPeriod;
	float absoluteRefractoryPeriodFraction;
	float contractionStrength;
	float4 color;
};

// This structure will contain all the switches that control the actions in the code.
struct simulationSwitchesStructure
{
	bool IsPaused;
	bool IsInAblateMode;
	bool IsInEctopicBeatMode;
	bool IsInEctopicEventMode;
	bool IsInAdjustMuscleAreaMode;
	bool IsInAdjustMuscleLineMode;
	bool IsInFindNodeMode;
	bool IsInMouseFunctionMode;
	bool MovieIsOn;
	// Turns the contractions on and off to speed up the simulation when only studying electrical activity.
	bool ContractionIsOn; 
	// 0 Orthogonal, 1 Frustum
	int ViewFlag; 
	// This is a three way toggle. With draw no nodes, draw the front half of the nodes, or draw all nodes.  0 = off, 1 = front half, 2 = all
	int DrawNodesFlag; 
	// Tells the program to draw the front half of the simulation or the full simulation.
	// We put it in because sometimes it is hard to tell if you are looking at the front of the simulation
	// or looking through a hole to the back of the simulation. By turning the back off it allows you to
	// orient yourself.
	int DrawFrontHalfFlag;
};

// Globals Start ******************************************

// This will hold all the nodes.
nodeAttributesStructure *Node;
nodeAttributesStructure *NodeGPU;

// This will hold all the muscles.
muscleAttributesStructure *Muscle;
muscleAttributesStructure *MuscleGPU;

// This will hold all the simulation switches.
simulationSwitchesStructure Switches;

// For videos and screenshots variables
FILE* MovieFile; // File that holds all the movie frames.
int* Buffer; // Buffer where you create each frame for a movie or the one frame for a screen shot.

// To setup your CUDA device
dim3 BlockNodes, GridNodes;
dim3 BlockMuscles, GridMuscles;

// This is the node that the beat initiates from.
int PulsePointNode;

// Nodes that orient the simulation. 
// If the node's center of mass is at <0,0,0> and the UpNode is up and FrontNode is in the front looking at you, you should be in the standard view.
int UpNode;
int FrontNode;

// Holds the name of view you are in for displaying in the terminal print.
char ViewName[256] = "no view set"; 

// These two variable get user input to adjust muscle refractory periods and conduction velocities when you are
// in AdjustMuscleAreaMode or AdjustMuscleLineMode modes. Once they are read in, they are multiplied by the muscles 
// refractory period and conduction velocity respectively.  
float RefractoryPeriodAdjustmentMultiplier;
float MuscleConductionVelocityAdjustmentMultiplier;

// These are all the globals that are read in from the simulationSetup file and are explained in detail there.
// simulationSetup globals start ************************************************
int NodesMusclesFileOrPreviousRunsFile;
char NodesMusclesFileName[256];
char PreviousRunFileName[256];
float LineWidth;
float NodeRadiusAdjustment;
double MyocyteLength; 
double MyocyteWidth;
double MyocyteContractionForce;
double MyocardialTissueDensity;
double MyocyteForcePerMassMultiplier;
double MyocyteForcePerMassSTD;
double DiastolicPressureLA;
double SystolicPressureLA;
double PressureMultiplier;
double MassOfLeftAtrium;
double VolumeOfLeftAtrium;
double Drag;
// bool ContractionIsOn; // This is read in from the setup file but is defined in the simulationSwitchesStructure.
double MuscleRelaxedStrengthFraction;
double MuscleCompressionStopFraction;
double MuscleCompressionStopFractionSTD;
double BaseMuscleRefractoryPeriod;
double MuscleRefractoryPeriodSTD;
double BaseAbsoluteRefractoryPeriodFraction;
double AbsoluteRefractoryPeriodFractionSTD;
double BaseMuscleConductionVelocity;
double MuscleConductionVelocitySTD;
double BeatPeriod;
double PrintRate;
int DrawRate;
double Dt;
float4 ReadyColor;
float4 ContractingColor;
float4 RestingColor;
float4 RelativeColor;
float4 DeadColor;
float4 BackGround;
// simulationSetup globals end ************************************************

// This will hold the radius of the left atrium which we will use to scale the size of everything in
// the simulation.
double RadiusOfLeftAtrium;

// This will hold the force per mass fraction of a myocte which we will use to scale a a muscles strength
// by its mass.
double MyocyteForcePerMassFraction;

// Variable that holds mouse locations to be translated into positions in the simulation.
double MouseX, MouseY, MouseZ;
int MouseWheelPos;
float HitMultiplier; // Adjusts how big of a region the mouse covers when you are selecting with it.
int ScrollSpeedToggle; // Sets slow or fast scroll speed.
double ScrollSpeed; // How fast your scroll moves.

// Times to keep track of what to do in the nBody() function and your progress through the simulation.
// Some of the variables that accompany this variable are read in from the simulationSetup file.
// The timers tell what the time is from the last action and the rates tell how often to perform the action.
double PrintTimer;
int DrawTimer; 
int RecenterCount;
int RecenterRate;
double RunTime;

// These keep track of where the view is as you zoom in and out and rotate.
float4 CenterOfSimulation;
float4 AngleOfSimulation;

// Window globals
static int Window;
int XWindowSize;
int YWindowSize; 
double Near; // Front and back of clip planes
double Far;
double EyeX; // Where your eye is
double EyeY;
double EyeZ;
double CenterX; // Where you are looking
double CenterY;
double CenterZ;
double UpX; // What up means to the viewer
double UpY;
double UpZ;

// How many nodes and muscle the simulation contains.
int NumberOfNodes;
int NumberOfMuscles;
	
// Prototyping functions start *****************************************************
// Functions in the SVT.h file.
void nBody(double);
void allocateMemory();
void readSimulationParameters();
void setup();
int main(int, char**);

// Functions in the CUDAFunctions.h file.
__device__ void turnOnNodeMusclesGPU(int, int, int, muscleAttributesStructure *, nodeAttributesStructure *);
__global__ void getForces(muscleAttributesStructure *, nodeAttributesStructure *, float, int, float4, float, float, float, float);
__global__ void updateNodes(nodeAttributesStructure *, int, int, muscleAttributesStructure *, float, float, float, bool);
__global__ void updateMuscles(muscleAttributesStructure *, nodeAttributesStructure *, int, int, float, float4, float4, float4, float4);
__global__ void recenter(nodeAttributesStructure *, int, float, float4);
void cudaErrorCheck(const char *, int);
void copyNodesMusclesToGPU();
void copyNodesMusclesFromGPU();

// Functions in the setNodesAndMuscles.h file.
void setNodesFromBlenderFile();
void checkNodes();
void setMusclesFromBlenderFile();
void linkNodesToMuscles();
double croppedRandomNumber(double, double, double);
void setRemainingNodeAndMuscleAttributes();
void getNodesandMusclesFromPreviuosRun();
void setRemainingParameters();
void hardCodedAblations();
void hardCodedPeriodicEctopicEvents();
void hardCodedIndividualMuscleAttributes();
void checkMuscle(int);
 
// Functions in the viewDrawAndTerminalFunctions.h file.
void orthogonalView();
void frustumView();
float4 findCenterOfMass();
void centerObject();
void rotateXAxis(float);
void rotateYAxis(float);
void rotateZAxis(float);
void ReferenceView();
void PAView();
void APView();
void setView(int);
void drawPicture();
void terminalPrint();
void helpMenu();

// Functions in the callBackFunctions.h file.
void Display(void);
void idle();
void reshape(int, int);
void mouseFunctionsOff();
void mouseAblateMode();
void mouseEctopicBeatMode();
void mouseAdjustMusclesAreaMode();
void mouseAdjustMusclesLineMode();
void mouseIdentifyNodeMode();
bool setMouseMuscleAttributes();
void setMouseMuscleRefractoryPeriod();
void setMouseMuscleConductionVelocity();
void setEctopicBeat(int);
void clearStdin();
void getEctopicBeatPeriod(int);
void getEctopicBeatOffset(int);
string getTimeStamp();
void movieOn();
void movieOff();
void screenShot();
void saveSettings();
void KeyPressed(unsigned char, int, int);
void mousePassiveMotionCallback(int, int);
void myMouse(int, int, int, int);

