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

// Cuda defines
#define BLOCKNODES 256
#define BLOCKMUSCLES 256
#define BLOCKCENTEROFMASS 512

// defines for terminal print
#define BOLD_ON  "\e[1m"
#define BOLD_OFF   "\e[m"

// Math defines.
#define PI 3.141592654

// Structure defines.
#define MUSCLES_PER_NODE 20

// Globals Start ******************************************

// For videos and screenshots
FILE* MovieFile;
int* Buffer;
bool MovieIsOn;

// To setup your CUDA device
dim3 BlockNodes, GridNodes;
dim3 BlockMuscles, GridMuscles;

// For Timing
float Dt;
float PrintRate;
int DrawRate;
int RecenterRate;
bool PauseIs;

// This is the node that the beat iminates from.
int PulsePointNode;

// Nodes that orient the simulation. 
// If UpNode is up and FrontNode is in the front you should be in the standard view.
int UpNode;
int FrontNode;

bool AblateModeIs;
bool EctopicBeatModeIs;
bool EctopicEventModeIs;
bool AdjustMuscleModeIs;
bool FindNodeModeIs;
bool MouseFunctionModeIs;
int ViewFlag; // 0 orthoganal, 1 fulstum

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
float MyocyteForcePerMassSTD;
float DiastolicPressureLA;
float SystolicPressureLA;
float PressureMultiplier;

float BeatPeriod;

float MassOfLeftAtrium;
float RadiusOfLeftAtrium;

int NumberOfNodes;
int NumberOfMuscles;

bool ContractionIsOn;
float MuscleRelaxedStrengthFraction;
float MuscleCompresionStopFraction;
float MuscleCompresionStopFractionSTD;
float BaseMuscleRefractoryPeriod;
float MuscleRefractoryPeriodSTD;
float RefractoryPeriodAdjustmentMultiplier;
float BaseMuscleConductionVelocity;
float MuscleConductionVelocitySTD;
float MuscleConductionVelocityAdjustmentMultiplier;
float BaseMuscleContractionStrength;
float AbsoluteRefractoryPeriodFraction;
float AbsoluteRefractoryPeriodFractionSTD;

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
	bool isOn;
	bool isDisabled;
	float timer;
	float mass;
	float naturalLength;
	float relaxedStrength;
	float compresionStopFraction;
	float conductionVelocity;
	float conductionDuration;
	float refractoryPeriod;
	float absoluteRefractoryPeriodFraction;
	float contractionStrength;
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
	bool isBeatNode;
	float beatPeriod;
	float beatTimer;
	bool isFiring;
	bool isAblated;
	bool drawNodeIs;
	float4 color;
	int muscle[MUSCLES_PER_NODE];
};

nodeAtributesStructure *Node;
nodeAtributesStructure *NodeGPU;

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
	
// Prototyping functions start *****************************************************
// Functions in the SVT.h file.
void n_body(float);
void allocateMemory();
void readSimulationParameters();
void setup();
int main(int, char**);

// Functions in the CUDAFunctions.h file.
__device__ void turnOnNodeMusclesGPU(int, int, int, muscleAtributesStructure *, nodeAtributesStructure *);
__global__ void getForces(muscleAtributesStructure *, nodeAtributesStructure *, float, int, float4, float, float, float, float);
__global__ void updateNodes(nodeAtributesStructure *, int, int, muscleAtributesStructure *, float, float, double, bool);
__global__ void updateMuscles(muscleAtributesStructure *, nodeAtributesStructure *, int, int, float, float4, float4, float4, float4);
__global__ void recenter(nodeAtributesStructure *, int, float4, float4);

void cudaErrorCheck(const char *, int);
void copyNodesMusclesToGPU();
void copyNodesMusclesFromGPU();

// Functions in the setNodesAndMuscles.h file.
void setNodesFromBlenderFile();
void checkNodes();
void setMusclesFromBlenderFile();
void linkNodesToMuscles();
double getLogNormal(float);
void setRemainingNodeAndMuscleAttributes();
void getNodesandMusclesFromPreviuosRun();
void checkMuscle(int);

// Functions in the hardCodedNodeAndMuscleAtributes.h file.
void hardCodedAblations();
void hardCodedPeriodicEctopicEvents();
void setIndividualMuscleAttributes();
 
// Functions in the DrawAndTerminalFunctions.h file.
void rotateXAxis(float);
void rotateYAxis(float);
void rotateZAxis(float);
void ReferenceView();
void PAView();
void APView();
void setView(int);
void drawPicture();
void terminalPrint();

// Functions in the callBackFunctions.h file.
void Display(void);
void idle();
void reshape(int, int);
void orthoganialView();
void fulstrumView();
void mouseFunctionsOff();
void mouseAblateMode();
void mouseEctopicBeatMode();
void mouseAdjustMusclesMode();
void mouseIdentifyNodeMode();
int setMouseMuscleAttributes();
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
void helpMenu();
void KeyPressed(unsigned char, int, int);
void mousePassiveMotionCallback(int, int);
void mymouse(int, int, int, int);
float4 findCenterOfMass();
void centerObject();

