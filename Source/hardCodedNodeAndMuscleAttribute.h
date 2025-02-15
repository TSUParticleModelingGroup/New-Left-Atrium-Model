/*
 This file contains functions where the user can hard code in node and muscle characteristics.
 These can all be done in a running simulation with the mouse but if you have a set that you 
 know you want to set at startup you can do them here.
 The functions are listed below in the order they appear.
 
 void hardCodedAblations();
 void hardCodedPeriodicEctopicEvents();
 void setIndividualMuscleAttributes();
*/

/*
 If you know that you want to ablated a set of nodes before the simulation
 starts you can do it here, or just wait and do it in the running simulation.
 Do not ablate the PulsePointNode node unless you want to have a simulation 
 that just sets there.
 
 An example is give and comented out below to work from.
*/
void hardCodedAblations()
{	
	// To ablate a slected node set your index and uncomment this line.
	
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
		printf("\n If this is what you wanted to do fine.");
		printf("\n If not change your sellection in the code hardCodedAblations() function.");
		printf("\n");
	}
	*/
}

/*
 If you know that you want to set a node to be a pulse node before the simulation
 starts you can do it here, or just wait and do it in the running simulation.
 Do not set the the PulsePointNode node because it has alread been set in the 
 setNodesFromBlenderFile() function
 
 An example is give and comented out below to work from.
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
 If you know that you want to set a muscle's atributes before the simulation
 starts you can do it here, or just wait and do it in the running simulation.
 
 An example is give and comented out below to work from.
*/
void setIndividualMuscleAttributes()
{
	/*
	int index = 100; // Set index to the muscle number you want.
	Muscle[index].conductionVelocity = BaseMuscleConductionVelocity*(10.0);
	Muscle[index].conductionDuration = Muscle[index].naturalLength/Muscle[index].conductionVelocity;
	Muscle[index].refractoryPeriod = BaseMuscleRefractoryPeriod*(10.0);
	checkMuscle(index);
	*/
}

