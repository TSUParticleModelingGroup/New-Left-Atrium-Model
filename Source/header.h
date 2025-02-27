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
struct nodeAttributesStructure //attributes is spelled wrong,change throughout- kyla
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
	bool drawNodeIs;
	float4 color;
	int muscle[MUSCLES_PER_NODE];
};

// Everything a muscle holds. We have 1 on the CPU and 1 on the GPU
struct muscleAttributesStructure //attributes is spelled wrong-kyla
{
	int nodeA;
	int nodeB;    
	int apNode;
	bool isOn;
	bool isDisabled;
	float timer;
	float mass;
	float naturalLength;
	float relaxedStrength;
	float compressionStopFraction; //compression is spelt wrong-kyla
	float conductionVelocity;
	float conductionDuration;
	float refractoryPeriod;
	float absoluteRefractoryPeriodFraction;
	float contractionStrength;
	float4 color;
};

// Globals Start ******************************************

// This will hold all the nodes.
nodeAttributesStructure *Node;
nodeAttributesStructure *NodeGPU;

// This will hold all the muscles.
muscleAttributesStructure *Muscle;
muscleAttributesStructure *MuscleGPU;

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

// These are the switches that tell what action you are performing in the simulation.
bool IsPaused;
bool AblateModeIs;
bool EctopicBeatModeIs;
bool EctopicEventModeIs;
bool AdjustMuscleAreaModeIs;
bool AdjustMuscleLineModeIs;
bool FindNodeModeIs;
bool MouseFunctionModeIs;
bool MovieIsOn;
int ViewFlag; // 0 orthogonal, 1 fulcrum (did you mean fulcrum-kyla?)

// This is a three way toggle. With draw no nodes, draw the front half of the nodes, or draw all nodes.  
int DrawNodesFlag;

// Tells the program to draw the front half of the simulation or the full simulation.
// We put it in because sometimes it is hard to tell if you are looking at the front of the simulation
// or looking through a hole to the back of the simulation. By turning the back off it allows you to
// orient yourself.
int DrawFrontHalfFlag;

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
float MyocyteForcePerMass;
float MyocyteForcePerMassMultiplier;
float MyocyteForcePerMassSTD;
float DiastolicPressureLA;
float SystolicPressureLA;
float PressureMultiplier;
float MassOfLeftAtrium;
float RadiusOfLeftAtrium;
float Drag;
bool ContractionIsOn;
float MuscleRelaxedStrengthFraction;
float MuscleCompressionStopFraction;
float MuscleCompressionStopFractionSTD;
float BaseMuscleRefractoryPeriod;
float MuscleRefractoryPeriodSTD;
float BaseAbsoluteRefractoryPeriodFraction;
float AbsoluteRefractoryPeriodFractionSTD;
float BaseMuscleConductionVelocity;
float MuscleConductionVelocitySTD;
float BeatPeriod;
float PrintRate;
int DrawRate;
float Dt;
float4 ReadyColor;
float4 ContractingColor;
float4 RestingColor;
float4 RelativeColor;
float4 DeadColor;
float4 BackGround;
// simulationSetup globals end ************************************************

// This is the base muscle strength that every muscle's start up strength is based on.
// You might think this should be read in from the simulationSetup file. The reason it is 
// not is because we do not know a muscles mass yet. Once we have the mass of a muscle,
// which we the calculated in 2: of the setRemainingNodeAndMuscleAttributes() function, we
// multiply it by the MyocyteForcePerMassMultiplier to get its base strength.
float BaseMuscleContractionStrength;

// Variable that holds mouse locations to be translated into positions in the simulation.
double MouseX, MouseY, MouseZ;
int MouseWheelPos;
float HitMultiplier; // Adjusts how big of a region the mouse covers when you are selecting with it.
int ScrollSpeedToggle; // Sets slow or fast scroll speed.
float ScrollSpeed; // How fast your scroll moves.

// Times to keep track of what to do in the nBody() function and your progress through the simulation.
// Some of the variables that accompany this variable are read in from the simulationSetup file.
// The timers tell what the time is from the last action and the rates tell how often to perform the action.
float PrintTimer;
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
void nBody(float);
void allocateMemory();
void readSimulationParameters();
void setup();
int main(int, char**);

// Functions in the CUDAFunctions.h file.
__device__ void turnOnNodeMusclesGPU(int, int, int, muscleAttributesStructure *, nodeAttributesStructure *);
__global__ void getForces(muscleAttributesStructure *, nodeAttributesStructure *, float, int, float4, float, float, float, float);
__global__ void updateNodes(nodeAttributesStructure *, int, int, muscleAttributesStructure *, float, float, double, bool);
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

