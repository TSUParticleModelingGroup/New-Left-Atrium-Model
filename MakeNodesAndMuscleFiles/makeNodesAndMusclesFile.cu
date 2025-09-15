// To compile the simulation type the line below in the terminal.
// nvcc makeNodesAndMusclesFile.cu -o temp
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <signal.h>
#include <unistd.h>
#include <stdbool.h>
#define PI 3.14

int Type = 2;
// A normal left atrium holds around 23,640 cubic millimeters of blood.
double Volume = 23640.0;
double Radius;
int NumberOfNodes;
int NumberOfMuscles;
float3* Node;
int2* Muscle;
int PulsePointNode;
int UpNode;
int FrontNode;

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

void setNodesAndMusclesLine() 
{
	NumberOfNodes = 10;
	NumberOfMuscles = NumberOfNodes - 1;
	
	PulsePointNode = 0;
	UpNode = 0;
	FrontNode = 0;
	
	float lengthOfLine = 2.0*PI*Radius;
	
	Node = (float3*)malloc(NumberOfNodes*sizeof(float3));
	Muscle = (int2*)malloc(NumberOfMuscles*sizeof(int2));
		
	// Setting the positions on a line.
	float start = -lengthOfLine/2.0;
	float dx = lengthOfLine/(NumberOfNodes - 1.0);
	for(int i = 0; i < NumberOfNodes; i++)
	{
		Node[i].x = start + i*dx;
		Node[i].y = 0.0;
		Node[i].z = 0.0;
	}
	
	// Setting the Muscles links to -1 so you can see if they are not used.
	for(int i = 0; i < NumberOfMuscles; i++)
	{
		Muscle[i].x =  -1;
		Muscle[i].y =  -1;	
	}
	
	// Setting the Muscles.
	for(int i = 0; i < NumberOfMuscles; i++)
	{
		Muscle[i].x =  i;
		Muscle[i].y =  i + 1;	
	}
}

void setNodesAndMusclesSheet() 
{
	int nodesX = 200;
	int nodesY = 200;
	NumberOfNodes = nodesX*nodesY;
	NumberOfMuscles = (nodesX - 1)*nodesY + (nodesY - 1)*nodesX;
	
	PulsePointNode = nodesX/2;
	UpNode = 0;
	FrontNode = 0;
	
	float width = 2.0*PI*Radius;
	float height = width*(float)nodesX/(float)nodesY;
	float dx =  width/(float)(nodesX - 1);
	float dy = height/(float)(nodesY - 1);
	double stddev = 0.5;
	
	Node = (float3*)malloc(NumberOfNodes*sizeof(float3));
	Muscle = (int2*)malloc(NumberOfMuscles*sizeof(int2));
		
	// Setting the positions on a line.
	float startX = -width/2.0;
	float startY = -height/2.0;
	int k = 0;
	for(int i = 0; i < nodesX; i++)
	{
		for(int j = 0; j < nodesX; j++)
		{
			Node[k].x = startX + i*dx + croppedRandomNumber(stddev, -dx/2.2, dx/2.2);
			Node[k].y = startY + j*dy + croppedRandomNumber(stddev, -dy/2.2, dy/2.2);;
			Node[k].z = 0.0;
			k++;
		}
	}
	
	// Setting the Muscles links to -1 so you can see if they are not used.
	for(int i = 0; i < NumberOfMuscles; i++)
	{
		Muscle[i].x =  -1;
		Muscle[i].y =  -1;	
	}
	
	// Setting the Muscles.
	k = 0;
	for(int j = 0; j < nodesY; j++)
	{
		for(int i = j*nodesX; i < (j+1)*nodesX; i++)
		{
			if(i < (j+1)*nodesX - 1)
			{
				Muscle[k].x =  i;
				Muscle[k].y =  i + 1;
				k++;
			}
			if(j < (nodesY - 1))
			{
				Muscle[k].x =  i;
				Muscle[k].y =  i + nodesX;
				k++;
			}
		}
	}
}

void saveNodesAndMuscle()
{
	const char *folderName = "Name It";
	
	// Creating the diretory to hold the run settings.
	if(mkdir(folderName, 0777) == 0)
	{
		printf("\n Directory '%s' created successfully.\n", folderName);
	}
	else
	{
		printf("\n Error creating directory '%s'.\n", folderName);
	}
	
	// Moving into the directory
	chdir(folderName);
	
	// Copying all the Nodes into this folder in the file named Nodes.
	FILE *NodesFile;
  	NodesFile = fopen("Nodes", "wb");
  	if (NodesFile == NULL) 
  	{
		printf("Error opening NodesFile!\n");
		exit(0);
    	}
    	fprintf(NodesFile, "%d\n", NumberOfNodes);
  	fprintf(NodesFile, "%d\n", PulsePointNode);
  	fprintf(NodesFile, "%d\n", UpNode);
  	fprintf(NodesFile, "%d\n", FrontNode);
  	
  	for(int i = 0; i < NumberOfNodes; i++)
	{
		fprintf(NodesFile,"%d %f %f %f\n", i, Node[i].x, Node[i].y, Node[i].z);
	}
  	fclose(NodesFile);
  	
  	// Copying all the muscules into this folder in the file named Muscles.
	FILE *MusclesFile;
  	MusclesFile = fopen("Muscles", "wb");
  	if (MusclesFile == NULL) 
  	{
		printf("Error opening MusclesFile!\n");
		exit(0);
    	}
	fprintf(MusclesFile, "%d\n", NumberOfMuscles);
  	for(int i = 0; i < NumberOfMuscles; i++)
	{
		fprintf(MusclesFile,"%d %d %d\n", i, Muscle[i].x, Muscle[i].y);
	}
  	fclose(MusclesFile);
  	
	// Moving back to original directory.
	chdir("../");

	printf("\n Nodes and Muscles saved.\n");
}

int main(int argc, char** argv)
{
	Radius = pow((3.0*Volume)/(4.0*PI), 1.0/3.0);
	if(Type == 1)
	{
		setNodesAndMusclesLine();
	}
	if(Type == 2)
	{
		setNodesAndMusclesSheet();
	}
	saveNodesAndMuscle();
	return 0;
}
