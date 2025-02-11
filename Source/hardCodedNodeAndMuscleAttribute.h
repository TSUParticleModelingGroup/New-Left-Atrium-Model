/*
 This file contains functions where the user can hard code in node and muscle characteristics.
 These can all be done in a running simulation with the mouse but if you have a set that you 
 know you want done at startup you can do them here.
 The functions are listed below in the order they appear.
 
 void hardCodedAblations();
 void hardCodedPeriodicEctopicEvents();
 void setIndividualMuscleAttributes();
*/

void hardCodedAblations()
{	
	// Note start and index must be lass than NumberOfNodes and stop most be less than or equal to NumberOfNodes.
	
	// To ablate a slected string of nodes set start and stop and uncomment this for loop.
	/*
	int start = ??;
	int stop = ??;
	for(int i = start; i < stop; i++)
	{	
		Node[i].ablatedYesNo = 1;
		Node[i].drawFlag = 1;
		Node[i].color.x = 1.0;
		Node[i].color.y = 1.0;
		Node[i].color.z = 1.0;
	}
	*/
	
	// To ablate a slected node set your index and uncomment this line.
	/*
	int index = ??;
	Node[index].ablatedYesNo = 1;
	Node[index].drawFlag = 1;
	Node[index].color.x = 1.0;
	Node[index].color.y = 1.0;
	Node[index].color.z = 1.0;
	*/
}

void hardCodedPeriodicEctopicEvents()
{	
	// This is the sinus beat.
	Node[PulsePointNode].beatNode = true;
	Node[PulsePointNode].beatPeriod = BeatPeriod;
	Node[PulsePointNode].beatTimer = BeatPeriod; // Set the time to BeatPeriod so it will kickoff a beat as soon as it starts.
	
	// To set a recurrent ectopic event set your index(node) and event(which ectopic event) and uncomment the following lines.
	// Note event must be less than the MaxNumberOfperiodicEctopicEvents and don't use event = 0, that is reserved for the sinus beat.
	/*
	event = ???; // Don't use 0 that is reserved for the sinus beat.
	int index = ???; 
	if(Node[index].ablatedYesNo != 1)
	{
		EctopicEvents[event].node = index;
		EctopicEvents[event].period = 10.0;
		EctopicEvents[event].time = EctopicEvents[event].period; // This will make it start right now.
		Node[index].drawFlag = 1;
		Node[index].color.x = 1.0;
		Node[index].color.y = 0.0;
		Node[index].color.z = 1.0;
	}
	*/
	
	// If you want to setup a random set of ectopic beats remove the ???s and uncomment these lines.
	/*
	int id;
	int numberOfRandomEctopicBeats = 0; // Must be less than the MaxNumberOfperiodicEctopicEvents.
	float beatPeriodUpperBound = 1000;
	time_t t;
	srand((unsigned) time(&t));
	
	if(MaxNumberOfperiodicEctopicEvents < numberOfRandomEctopicBeats)
	{
		printf("\n Your number of random beats is large than the total number of ectopic beats chosen in the setup file.");
		printf("\n Your number of random beats will be set to the max number of actopic beats");
		numberOfRandomEctopicBeats = MaxNumberOfperiodicEctopicEvents;
	}
	for(int i = 1; i < numberOfRandomEctopicBeats + 1; i++) // Must start at 1 because 0 is the sinus beat.
	{
		id = ((float)rand()/(float)RAND_MAX)*NumberOfNodes;
		EctopicEvents[i].node = id;
		Node[id].drawFlag = 1;
		if(Node[id].ablatedYesNo != 1)
		{
			Node[id].color.x = 0.69;
			Node[id].color.y = 0.15;
			Node[id].color.z = 1.0;
		}
		
		EctopicEvents[i].period = ((float)rand()/(float)RAND_MAX)*beatPeriodUpperBound;
		EctopicEvents[i].time = ((float)rand()/(float)RAND_MAX)*EctopicEvents[i].period;
		printf("\nectopic event %d node = %d period = %f, time = %f\n", i, EctopicEvents[i].node, EctopicEvents[i].period, EctopicEvents[i].time);
	}
	*/

}

void setIndividualMuscleAttributes()
{
	int start = 0;
	int stop = 0;
	int index = 0;
	
	if(NumberOfMuscles < start || NumberOfMuscles < stop || NumberOfMuscles < index)
	{
		printf("\n Stop or index is out of range.");
		printf("\n Good Bye \n");
		exit(0);
	}
	// To set individual muscles atribures follow the giuld below. This works on muscle index. Change the 1.0s to what ever you want.
	/*
	index = 1;
	Muscle[index].conductionVelocity = BaseMuscleConductionVelocity*(1.0);
	Muscle[index].conductionDuration = Muscle[index].naturalLength/Muscle[index].conductionVelocity;
	Muscle[index].contractionDuration = BaseMuscleContractionDuration*(1.0);
	Muscle[index].rechargeDuration = BaseMuscleRechargeDuration*(1.0);
	Muscle[index].contractionStrength = MyocyteForcePerMassMultiplier*MyocyteForcePerMass*Muscle[index].mass*(1.0);
	Muscle[index].relaxedStrength = BaseMuscleRelaxedStrengthFraction*Muscle[i].contractionStrength;
	*/
	// To change a sequential of muscles follow the guide below.
	/*
	for(int i = start; i < stop; i++)
	{
		Muscle[i].conductionVelocity = BaseMuscleConductionVelocity*(1.0);
		Muscle[i].conductionDuration = Muscle[i].naturalLength/Muscle[i].conductionVelocity;
		Muscle[i].contractionDuration = BaseMuscleContractionDuration*(1.0);
		Muscle[i].rechargeDuration = BaseMuscleRechargeDuration*(1.0);
		Muscle[i].contractionStrength = MyocyteForcePerMassMultiplier*MyocyteForcePerMass*Muscle[i].mass*(1.0);
		Muscle[i].relaxedStrength = BaseMuscleRelaxedStrengthFraction*Muscle[i].contractionStrength;
	}
	*/
	
	// Checking to see if the conduction wave leaves the muscle before it can reset.
	// If not a muscle could reset itself.
	float averageMuscleLength = 0;
	for(int i = 0; i < NumberOfMuscles; i++)
	{	
		if((Muscle[i].contractionDuration + Muscle[i].rechargeDuration) < Muscle[i].contractionDuration)
		{
		 	printf("\n Conduction duration is shorter than the (contraction plus recharge) duration in muscle number %d", i);
		 	printf("\nThis muscle will be disabled. \n");
		 	Muscle[i].disabled = true;
		 	Muscle[i].color.x = DeadColor.x;
			Muscle[i].color.y = DeadColor.y;
			Muscle[i].color.z = DeadColor.z;
			Muscle[i].color.w = 1.0;
		} 
		
		if(Muscle[i].contractionStrength < Muscle[i].relaxedStrength)
		{
		 	printf("\n The relaxed repultion strenrth of muscle %d is greater than its contraction strength. Rethink your parameters", i);
		 	printf("\n Good Bye \n");
		 	exit(0);
		} 
		
		averageMuscleLength += Muscle[i].naturalLength;
	}
	averageMuscleLength = averageMuscleLength/NumberOfMuscles;
	printf("\n The average muscle length = %f \n", averageMuscleLength);
	//exit(0);
}

