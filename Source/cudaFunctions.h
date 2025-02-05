/*
 This file contains all the CUDA and CUDA related function. 
 They are listed below in the order they appear.
 
 __device__ void turnOnNodeMusclesGPU(int, int, int, muscleAtributesStructure *, nodeAtributesStructure *, int *, ectopicEventStructure *, int);
 __global__ void getForces(muscleAtributesStructure *, nodeAtributesStructure *, int *, float dt, int, int, float4, float, float, float, float, int);
 __global__ void updateNodes(nodeAtributesStructure *, int, int, ectopicEventStructure *, int, muscleAtributesStructure *, int *, float, float, double, int);
 __global__ void updateMuscles(muscleAtributesStructure *, nodeAtributesStructure *, int *, ectopicEventStructure *, int, int, int, int, float, float4, float4, float4, float4, float);
 __global__ void recenter(nodeAtributesStructure *, int, float4, float4);
 void errorCheck(const char *);
 void cudaErrorCheck(const char *, int);
 void copyNodesMusclesToGPU();
 void copyNodesMusclesFromGPU();
*/


/*
 This CUDA function tries to turn on every muscle that is connected to a node.
 First it checks to see if this is a beat node. 
 ??? I'm not sure what to do withit if it is ???? I think we might should just return.
 Next it goes through all the muscle connected to that node, checks to see if it really 
 is a muscle, if it is it sees if it is off if it is it turns it on. There is no need to 
 see if a muscle is dead here because if it is dead turning it on will do nothing, so
 checking would just be a wasted check in the if statement.
*/
__device__ void turnOnNodeMusclesGPU(int nodeToTurnOn, int numberOfNodes, int linksPerNode, muscleAtributesStructure *muscle, nodeAtributesStructure *node, int *connectingMuscles, ectopicEventStructure *ectopicEvent, int maxNumberOfperiodicEctopicEvents)
{
	// Make sure that the AP duration is shorter than the contration+recharge duration or a muscle will turn itself on.
	int muscleNumber;
	if(node[nodeToTurnOn].ablatedYesNo != 1) // If the node is ablated just return.
	{
		for(int j = 0; j < maxNumberOfperiodicEctopicEvents; j++)
		{
			// Should we do this or just return. FInd out !!!!!!!!!! ????????????
			if(nodeToTurnOn == ectopicEvent[j].node)
			{
				ectopicEvent[j].time = 0.0; 
			}
		}
		
		for(int j = 0; j < linksPerNode; j++) // Looping through all muscle connected to this node.
		{
			if(numberOfNodes*linksPerNode <= (nodeToTurnOn*linksPerNode + j))
			{
				printf("\nTSU Error: number of ConnectingMuscles is out of bounds in turnOnNodeMusclesGPU\n");
			}
			muscleNumber = connectingMuscles[nodeToTurnOn*linksPerNode + j];
			
			if((muscleNumber != -1)) // Is this really a muscle. If it is -1 it is not.
			{
				if(muscle[muscleNumber].onOff == 0) // If muscle is off turn it on.
				{
					muscle[muscleNumber].apNode = nodeToTurnOn;  //This is the node where the AP wave will now start moving away from.
					muscle[muscleNumber].onOff = 1; // Set to on.
					muscle[muscleNumber].timer = 0.0; // Set timer.
				}
			}
		}
	}
}

/*
 This CUDA function calculates all of the position-forces on a node. Most of these forces are due 
 to the muscles connected to the nodes, but one is a central outward pushing force that represents 
 the pressure from the blood in the LA. 
 
 The only other force of a node is its drag force. This is a velocity-based force and is calculated 
 in the updateNodes function. 
 
 The function's parallel structure is node-based (GridNodes, BlockNodes).
 The different type forces are explained here and numbered so the reader can see where they are applied
 below in the function.
 
 1. Central push back force:
 	If you are using a lines or circle nodes and muscles file pressure does not make much sense but
 	this force is still usefull. In a lines simulation it helps straighten the line out after a beat.
 	In a circle simulation it helps return the nodes out to a circle after a beat.
 	For the 3-D shell simulations we use force = presure*area. The area of a node is calculated in the 
 	setMuscleAttributesAndNodeMasses() function. The pressure is is a linear function that starts from
 	DiastolicPressureLA when the node is a full radiusOfAtria and increases to SystolicPressureLA
 	when a node is at its contracted length of muscleCompresionStopFraction*radiusOfAtria. We also use 
 	a multilier so the user can adjust this force to fit their needs.
 
 This next set of functions gets the forces on a node caused by a muscles at all times not just when it
 is under contraction. It pulls back when a muscle is stretched past its natural length, pushes back when a 
 muscle is compressed past it maximal compression length, and helps restore a muscle to its natural length 
 when it is between these two values.
 
 2. Muscle is too short force: 
 	If a muscle starts to becomes shorter than compresionStopFraction*naturalLength it will start to push
 	back. To keep this from being an abrupt change we transition into it. This transition starts at 10% of 
 	the amount a muscle can contract with equals [0.1*(1 - compresionStopFraction)*naturalLength].
 	Note: We used the amount a muscle can contract not the compresionStopFraction because using the 
 	compresionStopFraction might make a case where the transition zone reaches past the natural length in
 	the subsequent force functions.
 	
 	This starts at the relaxed strength. The relaxed force is always on to help return a muscle to its natural 
 	length, as stated above. It cancels out the contraction strength by the time it hits the contraction stop length. 
	If d which equals the actual muscle length at this moment in time, gets shorter than the contraction stop length 
	the push back force just keeps increasing. This is acomplished in the linear function below 2..
 	
 3. Restoration force	
 	In this region the muscle is neather to short or too long. In this region we add a small push back force to help
 	the muscle return to its natural length. This is a constant forced force with strength equal to the relaxedStrength 
 	which is a fraction of the contractionStrength. This function and the pressure restore the AL to it relaxed shape.
 	
 4. Aproaching natural length transition force
 	As the muscle aproaches its natural length we linearly transition it into having a force of zero when it reaches 
 	its natural length.

 5. Muscle too long force
	If for some reason a muscle is stretched past its natural length this function will linearly pull it back to its 
	natural length. It should be a good bit stronger than the relaxedStrength. We are not putting in a distance where 
	the muscle actually brakes. If it gets stretched that far we have bigger problems we need to address. 
	We transition into this taking it from zero to the contractionStrength in one transitionLength. Past one transitionLength 
	it will just keep increasing. We use contractionStrength and transitionLength here because they are already scaled 
	to work on this muscle.
 	
 Below we create severl type contraction forces so the user can select what works best for them. They can use the simulationSetUp 
 file to select which one they prefer. Note: only one of these functions is run in a simulation. Also the user can completly turn
 these function if they only want to watch the electrical activity across the LA by setting the contractionType to zero in the
 simulationSetUp file.
 
 6. Constant contraction force
 	This contraction force just abruptly turns on with the full contractionStrength when the muscle is turned on. It then
 	abruptly turns off when the muscle timer goes past the half the refractory period to represent the relaxation phase. 
 	
 7. Linear contraction force				
	This contraction force transitions linearly from zero to full contractionStrength as the muscle timer goes from time 
	zero to half the refractory period. It then goes from full contractionStrength to zero as the muscle timer goes from
	half the refractory period to a full refractory period. 
	
 8. sinusoidal contraction force
 	This contraction force transitions from zero to full strength and back to zero as time goes through a full refractory
 	period using a sine function that has been adjusted to have a half period that matches the refractory period and an
 	amplitude that matchs the full contractionStrength.

 9. sine squared contraction force
 	This contraction force transitions from zero to full strength as time goes from zero to half the refractory period. 
 	Then goes from full strength to zero as time progresses on to the full refractory period. Here we use a sine squared 
 	function to acheive this.
 
*/
__global__ void getForces(muscleAtributesStructure *muscle, nodeAtributesStructure *node, int *connectingMuscles, float dt, int numberOfNodes, int linksPerNode, float4 centerOfSimulation, float muscleCompresionStopFraction, float radiusOfAtria, float diastolicPressureLA, float systolicPressureLA, int contractionType)
{
	float dx, dy, dz, d;
	int muscleNumber, nodeNumber;
	float x1,x2,y1,y2,m, transitionLength;
	float force;
	
	int i = threadIdx.x + blockDim.x*blockIdx.x; // This is the node number we will be working on.
	
	if(i < numberOfNodes) // checking to see if this node is in range.
	{
		// Zeroing out the forces on this node so we can start summing them.
		node[i].force.x   = 0.0;
		node[i].force.y   = 0.0;
		node[i].force.z   = 0.0;
		
		// 1. Central push back force
		dx = node[i].position.x - centerOfSimulation.x;
		dy = node[i].position.y - centerOfSimulation.y;
		dz = node[i].position.z - centerOfSimulation.z;
		d  = sqrt(dx*dx + dy*dy + dz*dz);
		if(0.00001 < d) // To keep from getting numeric overflow just jump over this if d is too small.
		{
			float r2 = muscleCompresionStopFraction*radiusOfAtria;
			m = (systolicPressureLA - diastolicPressureLA)/(r2 - radiusOfAtria);
			float bp = m*d + diastolicPressureLA - m*radiusOfAtria;
			force  = bp*node[i].area;
			node[i].force.x  += force*dx/d;
			node[i].force.y  += force*dy/d;
			node[i].force.z  += force*dz/d;
		}
		else
		{
			printf("\n TSU Error: Node %d has gotten really close to the center of the LA. Take a look at this!\n", i);
		}
		
		for(int j = 0; j < linksPerNode; j++) // Going through every muscle that is connected to the ith node.
		{
			muscleNumber = connectingMuscles[i*linksPerNode + j];
			if(muscleNumber != -1) // Checking to see if this is a valid muscle.
			{
				float contractionDuration = muscle[muscleNumber].contractionDuration;
				float rechargeDuration = muscle[muscleNumber].rechargeDuration;
				float totalDuration = contractionDuration + rechargeDuration;
				float timer = muscle[muscleNumber].timer;
				float contractionStrength = muscle[muscleNumber].contractionStrength;
				float relaxedStrength = muscle[muscleNumber].relaxedStrength;
				float naturalLength = muscle[muscleNumber].naturalLength;
				float compresionStopFraction = muscle[muscleNumber].compresionStopFraction;
				
				// Every muscle is connected to two nodes A and B. We know it is connected to the 
				// ith node. Now we need to find the node at the other end of the muscle.
				nodeNumber = muscle[muscleNumber].nodeA;
				// If the node number is yourself you must have the wrong end.
				if(nodeNumber == i) nodeNumber = muscle[muscleNumber].nodeB; 
			
				dx = node[nodeNumber].position.x - node[i].position.x;
				dy = node[nodeNumber].position.y - node[i].position.y;
				dz = node[nodeNumber].position.z - node[i].position.z;
				d  = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < 0.0001) // Grabbing numeric overflow before it happens.
				{
					printf("\n TSU Error: In generalMuscleForces d is very small between nodeNumbers %d and %d the seperation is %f. Take a look at this!\n", i, nodeNumber, d);
				}
				
				transitionLength = 0.1*(1.0 - compresionStopFraction)*naturalLength;
				// The following (2-5) force functions are always on.
				if(d < (compresionStopFraction*naturalLength + transitionLength))
				{
					// 2. Muscle is too short force
					x1 = compresionStopFraction*naturalLength;
					x2 = x1 + transitionLength;
					y1 = -contractionStrength;
					y2 = -relaxedStrength;
					m = (y2 - y1)/(x2 - x1);
					force = m*(d - x1) + y1;
				}
				else if(d < naturalLength - transitionLength)
				{
					// 3. Restoration force
					force = -relaxedStrength;
				}
				else if(d < naturalLength)
				{
					// 4. Aproaching natural length transition force
					m = relaxedStrength/transitionLength;
					force = m*(d - naturalLength);
				}
				else
				{
					// 5. Muscle too long force
					m = contractionStrength/transitionLength;
					force = m*(d - naturalLength - transitionLength) + contractionStrength;
				}
				node[i].force.x  += force*dx/d;
				node[i].force.y  += force*dy/d;
				node[i].force.z  += force*dz/d;
			
				// One of these functions will be turned on if the muscle is contracting.
				if(muscle[muscleNumber].onOff == 1 && muscle[muscleNumber].dead != 1)
				{
					// 6. Constant contraction force.
					if(contractionType == 1)
					{
						if(timer < contractionDuration)
						{
							force = contractionStrength;
							node[i].force.x += force*dx/d;
							node[i].force.y += force*dy/d;
							node[i].force.z += force*dz/d;
						}
					}
					
					// 7. Linear contraction force
					if(contractionType == 2) // Linear force
					{
						if(timer < contractionDuration) 
						{
							force = timer*(contractionStrength/contractionDuration);
							node[i].force.x += force*dx/d;
							node[i].force.y += force*dy/d;
							node[i].force.z += force*dz/d;
						}
		
						else if(timer < totalDuration)
						{
							force = (contractionStrength/rechargeDuration)*(totalDuration - timer);
							node[i].force.x += force*dx/d;
							node[i].force.y += force*dy/d;
							node[i].force.z += force*dz/d;
						}
						
					}
					
					// 8. sinusoidal contraction force
					if(contractionType == 3) // sine force
					{
						force = contractionStrength*sin(timer*PI/(totalDuration));
						node[i].force.x += force*dx/d;
						node[i].force.y += force*dy/d;
						node[i].force.z += force*dz/d;
					}
					
					// 9. sine squared contraction force
					if(contractionType == 4) // sine force
					{
					 	float temp = sin(timer*PI/(totalDuration));
						force = contractionStrength*temp*temp;
						node[i].force.x += force*dx/d;
						node[i].force.y += force*dy/d;
						node[i].force.z += force*dz/d;
					}
				}
				
			}
		}
	}
}

/*
 This CUDA function first moves the nodes then checks to see if the node is a beat node, if it is it updates its time 
 and if its time is past the beat period it sends out a signal then zeros out its timer to start a new period.
 
 Most of the forces on the node were calculated in the getForce GPU function but we need to get one more force.
 This force is based off of velocity and why we are doing it here. It is the drag force. When doing modeling you always 
 need to build in some way to remove excess energy from the system or the energy may build up and disturb your simulation.
 In nature this is usually experienced as heat and/or fliud resistance (drag). We are using a fluid drag force here to 
 remove energy biuld up. We are using a sphere moving through blood here because it felt relevant and we can scale it 
 to the problem but you could use any energy remove scheme you like as long as it is scaled to the problem.
 The drag force in a fluid is 
 F = (1/2)*c*p*A*v*v Where 
 c is the drag coefficient of the object: c for a sphere is 0.47
 p is the density of the fluid: p for blood is 1/1000 grams/mm^3
 v is the velocity of the object
 A is the area of the object facing the fluid.
 
 This force did not seem to be strong enough and the Atrium quivered and moved around a great deal. This might be 
 accurate if you placed an isolated beating atrium in a contaner of blood. In reality the atrium is conected to other 
 parts of the body which keep it in space. For our purposes we just needed it to remain in place a little better 
 so we added a multiplier so the user can adjust it in the simulation setup file.
*/
__global__ void updateNodes(nodeAtributesStructure *node, int numberOfNodes, int linksPerNode, ectopicEventStructure *ectopicEvent, int maxNumberOfperiodicEctopicEvents, muscleAtributesStructure *muscle, int *connectingMuscles, float dragMultiplier, float dt, double time, int contractionType)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(i < numberOfNodes)
	{
	
		if(contractionType != 0)
		{
			// Calculating the drag.
			float velocitySquared = node[i].velocity.x*node[i].velocity.x + node[i].velocity.y*node[i].velocity.y + node[i].velocity.z*node[i].velocity.z;
			float drag = dragMultiplier*node[i].area*0.000235*velocitySquared;
			
			// Moving the nodes forward in time with leap-frog.
			if(time == 0.0)
			{
				node[i].velocity.x += (node[i].force.x/node[i].mass - drag*node[i].velocity.x)*0.5*dt;
				node[i].velocity.y += (node[i].force.y/node[i].mass - drag*node[i].velocity.y)*0.5*dt;
				node[i].velocity.z += (node[i].force.z/node[i].mass - drag*node[i].velocity.z)*0.5*dt;
			}
			else
			{
				node[i].velocity.x += (node[i].force.x/node[i].mass - drag*node[i].velocity.x)*dt;
				node[i].velocity.y += (node[i].force.y/node[i].mass - drag*node[i].velocity.y)*dt;
				node[i].velocity.z += (node[i].force.z/node[i].mass - drag*node[i].velocity.z)*dt;
			}
			
			node[i].position.x += node[i].velocity.x*dt;
			node[i].position.y += node[i].velocity.y*dt;
			node[i].position.z += node[i].velocity.z*dt;
		}
		
		// Sending out a signal if the node is an beat node and the time is right.
		int j = 0;
		while(ectopicEvent[j].node != -1 && j < maxNumberOfperiodicEctopicEvents) // If it is -1 it is not set and no others after this will be set.
		{
			if(i == ectopicEvent[j].node)
			{
				ectopicEvent[j].time += dt;
				if(ectopicEvent[j].period < ectopicEvent[j].time)
				{
					turnOnNodeMusclesGPU(i, numberOfNodes, linksPerNode, muscle, node, connectingMuscles, ectopicEvent, maxNumberOfperiodicEctopicEvents);				
					ectopicEvent[j].time = 0.0;
					break;
				}
			}
			j++;
		}
	}	
}

/*
 This function triggers the next node when its signal reaches the end of the muscle.
 Then it colors the muscle depending on where the muscle is inits cycle.
 It a muscle reaches the end of its cycle it is turned off, its timer is set to zero,
 and its transmition direction set to undetermined by setting apNode to -1.
*/
__global__ void updateMuscles(muscleAtributesStructure *muscle, nodeAtributesStructure *node, int *connectingMuscles, ectopicEventStructure *ectopicEvent, int numberOfMuscles, int numberOfNodes, int linksPerNode, int maxNumberOfperiodicEctopicEvents, float dt, float4 readyColor, float4 contractingColor, float4 restingColor, float4 relativeColor, float relativeRefractoryPeriodFraction)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(i < numberOfMuscles)
	{
		if(muscle[i].onOff == 1 && muscle[i].dead != 1)
		{
			// Turning on the next node when the conduction front reaches it. This is at a certain floating point time this is why we used the +-dt
			// You can't just turn it on when the timer is greater than the conductionDuration because the timer is not rest here
			// and this would make this call happen every time step past conductionDuration until it was reset.
			if((muscle[i].conductionDuration - dt < muscle[i].timer) && (muscle[i].timer < muscle[i].conductionDuration + dt))
			{
				// Making the AP wave move forward through the muscle.
				if(muscle[i].apNode == muscle[i].nodeA)
				{
					turnOnNodeMusclesGPU(muscle[i].nodeB, numberOfNodes, linksPerNode, muscle, node, connectingMuscles, ectopicEvent, maxNumberOfperiodicEctopicEvents);
				}
				else
				{
					turnOnNodeMusclesGPU(muscle[i].nodeA, numberOfNodes, linksPerNode, muscle, node, connectingMuscles, ectopicEvent, maxNumberOfperiodicEctopicEvents);
				}
			}
			
			float refractoryPeriod = muscle[i].contractionDuration + muscle[i].rechargeDuration;
			float relativeRefractoryPeriod = refractoryPeriod*relativeRefractoryPeriodFraction;
			float absoluteRefractoryPeriod = refractoryPeriod - relativeRefractoryPeriod;
			
			if(muscle[i].timer < muscle[i].contractionDuration)
			{
				// Set color and update time.
				muscle[i].color.x = contractingColor.x; 
				muscle[i].color.y = contractingColor.y;
				muscle[i].color.z = contractingColor.z;
				muscle[i].timer += dt;
			}
			else if(muscle[i].timer < absoluteRefractoryPeriod)
			{ 
				// Set color and update time.
				muscle[i].color.x = restingColor.x;
				muscle[i].color.y = restingColor.y;
				muscle[i].color.z = restingColor.z;
				muscle[i].timer += dt;
			}
			else if(muscle[i].timer < refractoryPeriod)
			{ 
				// Set color and update time.
				muscle[i].color.x = relativeColor.x;
				muscle[i].color.y = relativeColor.y;
				muscle[i].color.z = relativeColor.z;
				muscle[i].timer += dt;
			}
			else
			{
				// There is no time update here just setting the color and turning the muscle off.
				muscle[i].color.x = readyColor.x;
				muscle[i].color.y = readyColor.y;
				muscle[i].color.z = readyColor.z;
				muscle[i].color.w = 1.0;
				
				muscle[i].onOff = 0;
				muscle[i].timer = 0.0;
				muscle[i].apNode = -1;
			}	
		}
	}	
}

/*
 Moves the center of mass of the nodes to the center of the simulation. The nodes tend to wonder because the model and
 the forces are not completely simetric. This function just moves it back to the center.
 We are doing this on one block so we do not have to jump out to sync the blocks then move the nodes to the center.
 Note: The block size here needs to be a power of 2 for the reduction to work. We check for this in the seto function in
 the SVT.cu file
 This scheme is working fine but I believe we could do this more eficently and also have it bring in the view you are in
 to improve the stablity of the user's view.
*/
__global__ void recenter(nodeAtributesStructure *node, int numberOfNodes, float4 centerOfMass, float4 centerOfSimulation)
{
	int id, n, nodeId;
	
	// This needs to be a power of two or the code will not work!!!
	__shared__ float4 myPart[BLOCKCENTEROFMASS];
	
	id = threadIdx.x;
	
	myPart[id].x = 0.0;
	myPart[id].y = 0.0;
	myPart[id].z = 0.0;
	myPart[id].w = 0.0;
	
	// Finding the number of strides needed to go through all of the nodes using only one block.
	int stop = (numberOfNodes - 1)/blockDim.x + 1;
	
	// Suming all of the node masses*distance and total node masses into the single block we are using.
	for(int i = 0; i < stop; i++)
	{
		nodeId = threadIdx.x + i*blockDim.x;
		// Checking to make sure we are not going past the number of nodes.
		// This will protect the sum if the number of nodes does not equally divide by the size of the block.
		if(nodeId < numberOfNodes)
		{
			myPart[id].x += node[nodeId].position.x*node[nodeId].mass;
			myPart[id].y += node[nodeId].position.y*node[nodeId].mass;
			myPart[id].z += node[nodeId].position.z*node[nodeId].mass;
			myPart[id].w += node[nodeId].mass;
		}
	}
	__syncthreads();
	
	// Doing the final reduction on the value we have acumulated on the block. 
	// This will all be stored in node zero when this while loop is done.
	// Note: This section of code only works if block size is a power of 2.
	n = blockDim.x;
	while(1 < n)
	{
		n /= 2;
		if(id < n)
		{
			myPart[id].x += myPart[id + n].x;
			myPart[id].y += myPart[id + n].y;
			myPart[id].z += myPart[id + n].z;
			myPart[id].w += myPart[id + n].w;
		}
		__syncthreads();
	}
	__syncthreads();
	
	// Dividing by the total mass will now give us the center of mass of all the nodes.
	if(id == 0)
	{
		myPart[0].x /= myPart[0].w;
		myPart[0].y /= myPart[0].w;
		myPart[0].z /= myPart[0].w;
	}
	__syncthreads();
	
	// Moving the nodes so that the nodes' center of mass will be at the center of the simulation.
	for(int i = 0; i < stop; i++)
	{
		nodeId = threadIdx.x + i*blockDim.x;
		if(nodeId < numberOfNodes)
		{
			node[nodeId].position.x -= myPart[0].x - centerOfSimulation.x;
			node[nodeId].position.y -= myPart[0].y - centerOfSimulation.y;
			node[nodeId].position.z -= myPart[0].z - centerOfSimulation.z;
		}	
	}
}

void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

/*
 Copies nodes and muscle files up to the GPU.
*/
void copyNodesMusclesToGPU()
{
	cudaMemcpy( MuscleGPU, Muscle, NumberOfMuscles*sizeof(muscleAtributesStructure), cudaMemcpyHostToDevice );
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy( NodeGPU, Node, NumberOfNodes*sizeof(nodeAtributesStructure), cudaMemcpyHostToDevice );
	cudaErrorCheck(__FILE__, __LINE__);
}

/*
 Copies nodes and muscle files down from the GPU.
*/
void copyNodesMusclesFromGPU()
{
	cudaMemcpy( Muscle, MuscleGPU, NumberOfMuscles*sizeof(muscleAtributesStructure), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy( Node, NodeGPU, NumberOfNodes*sizeof(nodeAtributesStructure), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
}
