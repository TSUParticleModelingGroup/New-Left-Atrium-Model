void errorCheck(const char *message)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: %s = %s\n", message, cudaGetErrorString(error));
		exit(0);
	}
}

__device__ void turnOnNodeMusclesGPU(int nodeToTurnOn, int NumberOfNodes, int linksPerNode, muscleAtributesStructure *Muscle, nodeAtributesStructure *Node, int *connectingMuscles, ectopicEventStructure *ectopicEvent, int maxNumberOfperiodicEctopicEvents)
{
	// Make sure that the AP duration is shorter than the contration+recharge duration or a muscle will turn itself on.
	int muscleNumber;
	if(Node[nodeToTurnOn].ablatedYesNo != 1) // If the node is ablated just return.
	{
		for(int j = 0; j < maxNumberOfperiodicEctopicEvents; j++)
		{
			if(nodeToTurnOn == ectopicEvent[j].node)
			{
				ectopicEvent[j].time = 0.0;
			}
		}
		
		for(int j = 0; j < linksPerNode; j++)
		{
			if(NumberOfNodes*linksPerNode <= (nodeToTurnOn*linksPerNode + j))
			{
				printf("\nTSU Error: number of ConnectingMuscles is out of bounds in turnOnNodeMusclesGPU\n");
			}
			muscleNumber = connectingMuscles[nodeToTurnOn*linksPerNode + j];
			
			if((muscleNumber != -1))
			{
				if(Muscle[muscleNumber].onOff == 0)
				{
					Muscle[muscleNumber].apNode = nodeToTurnOn;  //This is the node where the AP wave will now start moving away from.
					Muscle[muscleNumber].onOff = 1;
					Muscle[muscleNumber].timer = 0.0;
				}
			}
		}
	}
}

__global__ void getForces(muscleAtributesStructure *Muscle, nodeAtributesStructure *Node, int *ConnectingMuscles, float dt, int NumberOfNodes, int LinksPerNode, float4 CenterOfSimulation, float BaseMuscleCompresionStopFraction, float RadiusOfAtria, float BloodPressure, int contractionType)
{
	float dx, dy, dz, d;
	int muscleNumber, nodeNumber;
	float x1,x2,y1,y2,m, transitionLength;
	float force, velocity;
	
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(i < NumberOfNodes)
	{
		Node[i].force.x   = 0.0;
		Node[i].force.y   = 0.0;
		Node[i].force.z   = 0.0;
		
		// Getting forces on the nodes from the pressure of the blood pushing out. 
		// If you are using the line: nodes and muscles file,
		// there is no pressure on a line but this will just push it out along the x axis.
		// Also the there is no presure when using the circle: nodes and muscle file,
		// this will just push the nodes out from the center of the circle.
		// For the other node and muscle files it is just force = presure*area.
		dx = Node[i].position.x - CenterOfSimulation.x;
		dy = Node[i].position.y - CenterOfSimulation.y;
		dz = Node[i].position.z - CenterOfSimulation.z;
		d  = sqrt(dx*dx + dy*dy + dz*dz);
		// To keep from getting numeric overflow just jump over this if d is too small.
		if(0.00001 < d) 
		{
			force  = BloodPressure*Node[i].area;
			Node[i].force.x  += force*dx/d;
			Node[i].force.y  += force*dy/d;
			Node[i].force.z  += force*dz/d;
		}
		
		// When done modeling you always need to build in some way to remove 
		// excess energy from the system or the energy may build up and disturb your simulation.
		// In nature this is usually experienced as heat and/or fliud resistance (drag).
		// We are using a fluid drag force here to remove energy biuld up.
		// We are using a sphere moving through blood here because it felt relevant
		// and we can scale it to the problem but you could use any energy remove scheme you 
		// like as long as it is scaled to the problem.   
		// 
		// The drag force in a fluid is F = (1/2)*c*p*A*v*v
		// Where 
		// c is the drag coefficient of the object: c for a sphere is 0.47
		// p is the density of the fluid: p for blood is 1/1000 grams/mm^3
		// v is the velocity of the object
		// A is the area of the object facing the fluid.
		// This force did not seem to be strong enough and the Atrium quivered 
		// and moved around a great deal. This might be accurate if you placed 
		// an isolated beating atrium in a contaner of blood. In reality the atrium 
		// is conected to other parts of the body which keep it in space.
		// For our purposes we just needed it to remain in place a little better 
		// so we removed the (1/2)*c*p from the force function.
		// This added enough resistance so that the models' center of mass and orientation
		// moved around much less.
		velocity = sqrt(Node[i].velocity.x*Node[i].velocity.x + Node[i].velocity.y*Node[i].velocity.y + Node[i].velocity.z*Node[i].velocity.z);
		// Checking to make sure the node is moving or you would get a division by zero.
		if(0.00001 < velocity)
		{
			//float cc = 0.47;
			//float pp = 0.001;
			//force = Node[i].area*0.5*cc*pp*velocity*velocity;
			force = Node[i].area*velocity*velocity;
			
			// Dispersing the force relative to the velocity vector.
			Node[i].force.x   += -force*Node[i].velocity.x/velocity;
			Node[i].force.y   += -force*Node[i].velocity.y/velocity;
			Node[i].force.z   += -force*Node[i].velocity.z/velocity;
		}
		
		// Now we are getting the node to node forces caused by their connecting muscles.
		for(int j = 0; j < LinksPerNode; j++)
		{
			// This is the number of the muscles that connect node "i" to node "nodeNumber".
			muscleNumber = ConnectingMuscles[i*LinksPerNode + j];
			if(muscleNumber != -1)
			{
				float contractionDuration = Muscle[muscleNumber].contractionDuration;
				float rechargeDuration = Muscle[muscleNumber].rechargeDuration;
				float totalDuration = contractionDuration + rechargeDuration;
				float timer = Muscle[muscleNumber].timer;
				float contractionStrength = Muscle[muscleNumber].contractionStrength;
				float relaxedStrength = Muscle[muscleNumber].relaxedStrength;
				float naturalLength = Muscle[muscleNumber].naturalLength;
				float compresionStopFraction = Muscle[muscleNumber].compresionStopFraction;
				
				// This is the node number to work against. "i" is me "nodeNumber" is you. (That will make sense if you had me for DE)
				nodeNumber = Muscle[muscleNumber].nodeA;
				if(nodeNumber == i) nodeNumber = Muscle[muscleNumber].nodeB;
			
				dx = Node[nodeNumber].position.x - Node[i].position.x;
				dy = Node[nodeNumber].position.y - Node[i].position.y;
				dz = Node[nodeNumber].position.z - Node[i].position.z;
				d  = sqrt(dx*dx + dy*dy + dz*dz);
				// Grabbing numeric overflow before it happens.
				if(d < 0.0001) 
				{
					printf("\n TSU Error: In generalMuscleForces d is very small between nodeNumbers %d and %d the seperation is %f. Take a look at this!\n", i, nodeNumber, d);
				}
				
				// Getting forces on the nodes from the muscle fiber seperate from the contraction force.
				//
				// The first if statement addresses what hapens if a muslce starts to get shorter than it
				// should "less than the compresionStopFraction*naturalLength". To keep this from being an abrupt
				// change we transition into it. This transition starts at 10% of the amount a muscle can contract
				// "(1 - compresionStopFraction)*naturalLength". We used the amount a muscle can contract 
				// not the compresionStopFraction because using the compresionStopFraction might make a case 
				// where the transition zone reaches past the natural length. 
				transitionLength = 0.1*(1.0 - compresionStopFraction)*naturalLength;
				if(d < (compresionStopFraction*naturalLength + transitionLength))
				{
					// This starts at the relaxed strength (the relaxed force is always on to help return a muscle to its
					// natural length.) and cancels out the contraction strength by the time it hits the contraction stop length. 
					// If d (the actual muscle length at this moment in time) gets shorter than the contraction stop length the 
					// push back force just keeps increasing.
					x1 = compresionStopFraction*naturalLength;
					x2 = x1 + transitionLength;
					y1 = -contractionStrength;
					y2 = -relaxedStrength;
					m = (y2 - y1)/(x2 - x1);
					force = m*(d - x1) + y1;
				}
				else if(d < naturalLength - transitionLength)
				{
					// In here the muscle should do nothing if it is not contracting and if it is contracting we should just 
					// get out of the way and let it contract. But, we give it a little push back to help it return to its 
					// natural length when it is not contracting. It is a small proportion of the contraction strength
					// so the contraction force just over powers it during a contraction.
					// In short this push back force is always on.
					force = -relaxedStrength;
				}
				else if(d < naturalLength)
				{
					// As the muscle aproaches its natural length we transition it into having a force of zero on its node.
					// The function below take the muscle from have a push back of relaxedStrength to zero in one transition length.
					m = relaxedStrength/transitionLength;
					force = m*(d - naturalLength);
				}
				else
				{
					// If for some reason the muscle is stretched past its natural length this part of the function will pull 
					// it back to its natural length. It should be pretty strong. We are not putting in a distance where the muscle
					// actually brakes. If it gets stretched that far we have bigger problems we need to address.
					// We will transition into this taking it from zero to the contractionStrength in one transitionLength. Past 
					// one transitionLength it will just keep increasing. We use contractionStrength and transitionLength here 
					// because they are already scaled to work on this muscle.
					m = contractionStrength/transitionLength;
					force = m*(d - naturalLength - transitionLength) + contractionStrength;
				}
				Node[i].force.x  += force*dx/d;
				Node[i].force.y  += force*dy/d;
				Node[i].force.z  += force*dz/d;
			
				// Getting forces on the nodes from the muscle fibers that are under contraction.
				if(Muscle[muscleNumber].onOff == 1 && Muscle[muscleNumber].dead != 1)
				{
					// This type contraction just abruptly turns on the contration force 
					// in full when it is turned on and abruptly turns it off when the
					// Timer goes past the contraction time.
					if(contractionType == 1) // Constant force
					{
						if(timer < contractionDuration)
						{
							force = contractionStrength;
							Node[i].force.x += force*dx/d;
							Node[i].force.y += force*dy/d;
							Node[i].force.z += force*dz/d;
						}
					}
					
					// This type contraction transitions linearly from zero to full strength  
					// from time zero to contractionDuration. Then linearly transitions from 
					// full strength to zero as time goes from contractionDuration to the full
					// refractory period (totalDuration). 
					if(contractionType == 2) // Linear force
					{
						if(timer < contractionDuration) 
						{
							force = timer*(contractionStrength/contractionDuration);
							Node[i].force.x += force*dx/d;
							Node[i].force.y += force*dy/d;
							Node[i].force.z += force*dz/d;
						}
		
						else if(timer < totalDuration)
						{
							force = (contractionStrength/rechargeDuration)*(totalDuration - timer);
							Node[i].force.x += force*dx/d;
							Node[i].force.y += force*dy/d;
							Node[i].force.z += force*dz/d;
						}
						
					}
					
					// This type contraction transitions from zero to full strength as time
					// goes from zero to half the refractory period (totalDuration). Then goes from 
					// full strength to zero as time goes to the full refractory period (totalDuration).
					// This is basically the same as type 2 above but it uses a cosine function to create 
					// the transition.
					if(contractionType == 3) // cosine force
					{
						force = contractionStrength*sin(timer*PI/(totalDuration));
						Node[i].force.x += force*dx/d;
						Node[i].force.y += force*dy/d;
						Node[i].force.z += force*dz/d;
					}
				}
				
			}
		}
	}
}

__global__ void updateNodes(nodeAtributesStructure *node, int numberOfNodes, int linksPerNode, ectopicEventStructure *ectopicEvent, int maxNumberOfperiodicEctopicEvents, muscleAtributesStructure *muscle, int *connectingMuscles, float dt, double time)  // LeapFrog
{
	// Moving the nodes forward in time with leap-frog.
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(i < numberOfNodes)
	{
		if(time == 0.0)
		{
			node[i].velocity.x += (node[i].force.x/node[i].mass)*0.5*dt;
			node[i].velocity.y += (node[i].force.y/node[i].mass)*0.5*dt;
			node[i].velocity.z += (node[i].force.z/node[i].mass)*0.5*dt;
		}
		else
		{
			node[i].velocity.x += (node[i].force.x/node[i].mass)*dt;
			node[i].velocity.y += (node[i].force.y/node[i].mass)*dt;
			node[i].velocity.z += (node[i].force.z/node[i].mass)*dt;
		}
		node[i].position.x += node[i].velocity.x*dt;
		node[i].position.y += node[i].velocity.y*dt;
		node[i].position.z += node[i].velocity.z*dt;
		
		for(int j = 0; j < maxNumberOfperiodicEctopicEvents; j++)
		{
			if(i == ectopicEvent[j].node)
			{
				ectopicEvent[j].time += dt;
				if(ectopicEvent[j].period < ectopicEvent[j].time)
				{
					turnOnNodeMusclesGPU(i, numberOfNodes, linksPerNode, muscle, node, connectingMuscles, ectopicEvent, maxNumberOfperiodicEctopicEvents);				
					ectopicEvent[j].time = 0.0;
				}
			}
		}
	}	
}

__global__ void updateMuscles(muscleAtributesStructure *Muscle, nodeAtributesStructure *Node, int *ConnectingMuscles, ectopicEventStructure *ectopicEvent, int NumberOfMuscles, int NumberOfNodes, int LinksPerNode, int maxNumberOfperiodicEctopicEvents, float dt, float4 readyColor, float4 contractingColor, float4 restingColor, float4 relativeColor, float percentRelativeRefractoryPeriod)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	if(i < NumberOfMuscles)
	{
		if(Muscle[i].onOff == 1 && Muscle[i].dead != 1)
		{
			// Turning on the next node when the conduction front reaches it. This is at a certain floating point time this is why we used the +-dt
			if((Muscle[i].conductionDuration - dt < Muscle[i].timer) && (Muscle[i].timer < Muscle[i].conductionDuration + dt))
			{
				// Making the AP wave move forward through the muscle.
				if(Muscle[i].apNode == Muscle[i].nodeA)
				{
					turnOnNodeMusclesGPU(Muscle[i].nodeB, NumberOfNodes, LinksPerNode, Muscle, Node, ConnectingMuscles, ectopicEvent, maxNumberOfperiodicEctopicEvents);
				}
				else
				{
					turnOnNodeMusclesGPU(Muscle[i].nodeA, NumberOfNodes, LinksPerNode, Muscle, Node, ConnectingMuscles, ectopicEvent, maxNumberOfperiodicEctopicEvents);
				}
			}
			
			float refractoryPeriod = Muscle[i].contractionDuration + Muscle[i].rechargeDuration;
			float relativeRefractoryPeriod = refractoryPeriod*percentRelativeRefractoryPeriod;
			float absoluteRefractoryPeriod = refractoryPeriod - relativeRefractoryPeriod;
			
			if(Muscle[i].timer < Muscle[i].contractionDuration)
			{
				// Set color and update time.
				Muscle[i].color.x = contractingColor.x; 
				Muscle[i].color.y = contractingColor.y;
				Muscle[i].color.z = contractingColor.z;
				Muscle[i].timer += dt;
			}
			else if(Muscle[i].timer < absoluteRefractoryPeriod)
			{ 
				// Set color and update time.
				Muscle[i].color.x = restingColor.x;
				Muscle[i].color.y = restingColor.y;
				Muscle[i].color.z = restingColor.z;
				Muscle[i].timer += dt;
			}
			else if(Muscle[i].timer < refractoryPeriod)
			{ 
				// Set color and update time.
				Muscle[i].color.x = relativeColor.x;
				Muscle[i].color.y = relativeColor.y;
				Muscle[i].color.z = relativeColor.z;
				Muscle[i].timer += dt;
			}
			else
			{
				// There is no time update here just setting the color and turning the muscle off.
				Muscle[i].color.x = readyColor.x;
				Muscle[i].color.y = readyColor.y;
				Muscle[i].color.z = readyColor.z;
				Muscle[i].color.w = 1.0;
				
				Muscle[i].onOff = 0;
				Muscle[i].timer = 0.0;
				Muscle[i].apNode = -1;
			}	
		}
	}	
}

__global__ void recenter(nodeAtributesStructure *node, int numberOfNodes, float4 centerOfMass, float4 centerOfSimulation)
{
	int id, n, nodeId;
	__shared__ float4 myPart[BLOCKNODES];
	
	if(BLOCKNODES != blockDim.x) 
	{
		printf("\n Error BLOCKNODES not equal to blockDim.x %d  %d", BLOCKNODES, blockDim.x);
		//exit(0);
	}
	
	id = threadIdx.x;
	
	myPart[id].x = 0.0;
	myPart[id].y = 0.0;
	myPart[id].z = 0.0;
	myPart[id].w = 0.0;
	
	int stop = (numberOfNodes - 1)/blockDim.x + 1;
	
	for(int i = 0; i < stop; i++)
	{
		nodeId = threadIdx.x + i*blockDim.x;
		if(nodeId < numberOfNodes)
		{
			myPart[id].x += node[nodeId].position.x*node[nodeId].mass;
			myPart[id].y += node[nodeId].position.y*node[nodeId].mass;
			myPart[id].z += node[nodeId].position.z*node[nodeId].mass;
			myPart[id].w += node[nodeId].mass;
		}
	}
	__syncthreads();
	
	n = blockDim.x;
	while(2 < n)
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
	
	if(id == 0)
	{
		myPart[0].x /= myPart[0].w;
		myPart[0].y /= myPart[0].w;
		myPart[0].z /= myPart[0].w;
	}
	__syncthreads();
	
	// Moving the center of mass to the center of the simulation.
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
