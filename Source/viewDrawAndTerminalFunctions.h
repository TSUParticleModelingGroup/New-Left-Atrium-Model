/*
 This file contains:
 1: All the functions that determine how to orient and view the simulation.
 2: all the functions that draw the actual simulation. 
 3: The functions that print to the linux terminal all the setting of the simulation.
 In short this file holds the functions that present information to the user.
 
 The functions are listed below in the order they appear.
 void renderSphere(float, int, int);
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
*/

/*
 This function sets your view to orthogonal. In orthogonal view all object are kept in line in the z direction.
 This is not how your eye sees things but can be useful when determining if objects are lined up along the z-axis. 
*/

// Add this to a utility file
void renderSphere(float radius, int slices, int stacks) 
{
    // Sphere geometry parameters
    float x, y, z, alpha, beta; // Storage for coordinates and angles
    float sliceStep = 2.0f * PI / slices;
    float stackStep = PI / stacks;

    for (int i = 0; i < stacks; ++i) {
        alpha = i * stackStep;
        beta = alpha + stackStep;

        glBegin(GL_TRIANGLE_STRIP);
        for (int j = 0; j <= slices; ++j) {
            float theta = (j == slices) ? 0.0f : j * sliceStep;

            // Vertex 1
            x = -sin(alpha) * cos(theta);
            y = cos(alpha);
            z = sin(alpha) * sin(theta);
            glNormal3f(x, y, z);
            glVertex3f(x * radius, y * radius, z * radius);

            // Vertex 2
            x = -sin(beta) * cos(theta);
            y = cos(beta);
            z = sin(beta) * sin(theta);
            glNormal3f(x, y, z);
            glVertex3f(x * radius, y * radius, z * radius);
        }
        glEnd();
    }
}

void orthogonalView()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-RadiusOfLeftAtrium, RadiusOfLeftAtrium, -RadiusOfLeftAtrium, RadiusOfLeftAtrium, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	Simulation.ViewFlag = 0;
	drawPicture();
}

/*
 This function sets your view to frustum.This is the view the your eyes actually see. Where train tracks pull in 
 towards each other as they move off in the distance. It is how we see but can cause problems when using the mouse
 which lives in 2D to locate an object that lives in 3D.
*/
void frustumView()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	Simulation.ViewFlag = 1;
	drawPicture();
}

/*
 This function finds the center of mass of the LA. It may seem like this function does not belong with
 the display functions but it is used to center th LA in the view.
*/
float4 findCenterOfMass()
{
	float4 centerOfMass;
	
	centerOfMass.x = 0.0;
	centerOfMass.y = 0.0;
	centerOfMass.z = 0.0;
	centerOfMass.w = 0.0;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		 centerOfMass.x += Node[i].position.x*Node[i].mass;
		 centerOfMass.y += Node[i].position.y*Node[i].mass;
		 centerOfMass.z += Node[i].position.z*Node[i].mass;
		 centerOfMass.w += Node[i].mass;
	}
	if(centerOfMass.w < 0.00001) // .w holds the mass.
	{
		printf("\n Mass is too small\n");
		printf("\nw Good Bye\n");
		exit(0);
	}
	else
	{
		centerOfMass.x /= centerOfMass.w;
		centerOfMass.y /= centerOfMass.w;
		centerOfMass.z /= centerOfMass.w;
	}
	return(centerOfMass);
}

/*
 This function centers the LA and resets the center of view to (0, 0, 0).
 It is called periodically in a running simulation to center the LA, because the LA is not symmetrical 
 and will wander off over time. It is also use center the LA before all the views are set.
*/
void centerObject()
{
	float4 centerOfMass = findCenterOfMass();
	for(int i = 0; i < NumberOfNodes; i++)
	{
		Node[i].position.x -= centerOfMass.x;
		Node[i].position.y -= centerOfMass.y;
		Node[i].position.z -= centerOfMass.z;
		
		//Node[i].velocity.x = 0.0;
		//Node[i].velocity.y = 0.0;
		//Node[i].velocity.z = 0.0;
	}
	CenterOfSimulation.x = 0.0;
	CenterOfSimulation.y = 0.0;
	CenterOfSimulation.z = 0.0;
}

/*
 This function rotates the view around the x-axis.
*/
void rotateXAxis(float angle)
{
	float temp;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp = cos(angle)*Node[i].position.y - sin(angle)*Node[i].position.z;
		Node[i].position.z  = sin(angle)*Node[i].position.y + cos(angle)*Node[i].position.z;
		Node[i].position.y  = temp;
	}
	AngleOfSimulation.x += angle;
}

/*
 This function rotates the view around the y-axis.
*/
void rotateYAxis(float angle)
{
	float temp;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp =  cos(-angle)*Node[i].position.x + sin(-angle)*Node[i].position.z;
		Node[i].position.z  = -sin(-angle)*Node[i].position.x + cos(-angle)*Node[i].position.z;
		Node[i].position.x  = temp;
	}
	AngleOfSimulation.y += angle;
}

/*
 This function rotates the view around the z-axis.
*/
void rotateZAxis(float angle)
{
	float temp;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp = cos(angle)*Node[i].position.x - sin(angle)*Node[i].position.y;
		Node[i].position.y  = sin(angle)*Node[i].position.x + cos(angle)*Node[i].position.y;
		Node[i].position.x  = temp;
	}
	AngleOfSimulation.z += angle;
}

/*
 This function puts the viewer in the reference view. The reference view is looking straight at the four
 pulmonary veins with a vein in each of the four quadrants of the x-y plane as symmetric as you can make 
 it with the mitral valve down. We base all the other views off of this view.
*/
void ReferenceView()
{	
	float angle, temp;
	
	centerObject();
		
	// Rotating until the up Node is on x-y plane above or below the positive x-axis.
	angle = atan(Node[UpNode].position.z/Node[UpNode].position.x);
	if(Node[UpNode].position.x < 0.0) angle -= PI;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp = cos(angle)*Node[i].position.x + sin(angle)*Node[i].position.z;
		Node[i].position.z  = -sin(angle)*Node[i].position.x + cos(angle)*Node[i].position.z;
		Node[i].position.x  = temp;
	}
	AngleOfSimulation.y += angle;
	
	// Rotating until up Node is on the positive y axis.
	angle = PI/2.0 - atan(Node[UpNode].position.y/Node[UpNode].position.x);
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp = cos(angle)*Node[i].position.x - sin(angle)*Node[i].position.y;
		Node[i].position.y  = sin(angle)*Node[i].position.x + cos(angle)*Node[i].position.y;
		Node[i].position.x  = temp;
	}
	AngleOfSimulation.z += angle;
	
	// Rotating until front Node is on the positive z axis.
	angle = atan(Node[FrontNode].position.z/Node[FrontNode].position.x) - PI/2.0;
	if(Node[FrontNode].position.x < 0.0) angle -= PI;
	for(int i = 0; i < NumberOfNodes; i++)
	{
		temp = cos(angle)*Node[i].position.x + sin(angle)*Node[i].position.z;
		Node[i].position.z  = -sin(angle)*Node[i].position.x + cos(angle)*Node[i].position.z;
		Node[i].position.x  = temp;
	}
	AngleOfSimulation.y += angle;
}

/*
 This function puts the LA in the PA view.
 The heart does not set in the chest at a straight on angle. Hence we need to adjust our 
 reference view to what is actually seen in a back view looking through the chest.
*/
void PAView()
{  
	float angle;
	
	ReferenceView();
	
	angle = PI/6.0; // Rotate 30 degrees counterclockwise on the y-axis 
	rotateYAxis(angle);

	angle = PI/6.0; // Rotate 30 degrees counterclockwise on the z-axis
	rotateZAxis(angle);
}

/*
 This function puts the LA in the AP view.
 To get the AP view we just rotate the PA view 180 degrees on the y-axis
*/
void APView()
{ 
	float angle;
	
	PAView();
	angle = PI; // Rotate 180 degrees counterclockwise on the y-axis 
	rotateYAxis(angle);
}

/*
 This function sets all the views based off of the reference view and the AP view.
*/
void setView(int view)
{
    if(view == 6)
    {
        ReferenceView();
        strcpy(ViewName, "Ref");
    }
    else if(view == 4)
    {
        PAView();
        strcpy(ViewName, "PA");
    }
    else if(view == 2)
    {
        APView();
        strcpy(ViewName, "AP");
    }
    else if(view == 3)
    {
        APView();
        rotateYAxis(-PI/6.0);
        strcpy(ViewName, "RAO");
    }
    else if(view == 1)
    {
        APView();
        rotateYAxis(PI/3.0);
        strcpy(ViewName, "LAO");
    }
    else if(view == 7)
    {
        APView();
        rotateYAxis(PI/2.0);
        strcpy(ViewName, "LL");
    }
    else if(view == 9)
    {
        APView();
        rotateYAxis(-PI/2.0);
        strcpy(ViewName, "RL");
    }
    else if(view == 8)
    {
        APView();
        rotateXAxis(PI/2.0);
        strcpy(ViewName, "SUP");
    }
    else if(view == 5)
    {
        APView();
        rotateXAxis(-PI/2.0);
        strcpy(ViewName, "INF");
    }
    else
    {
        printf("\n Undefined view reverting back to Ref view.");
    }
}

/*
 This function draws the LA to the screen. It also saves movie frames if a movie is being recorded.
*/
void drawPicture()
{
	//int nodeNumber;
	int muscleNumber;
	int k;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	// Draw Nodes

	// Drawing Pulse node
	if(!Simulation.isPaused) glColor3d(0.0,1.0,0.0);
	else glColor3d(1.0,0.0,0.0);

	glPushMatrix();
	glTranslatef(Node[PulsePointNode].position.x, Node[PulsePointNode].position.y, Node[PulsePointNode].position.z);
	renderSphere(0.03*RadiusOfLeftAtrium,20,20);
	glPopMatrix();
	
	// Drawing center node
	//This draws a center node at the center of the simulation for debugging purposes
	if(false) // false turns it off, true turns it on.
	{
		glColor3d(1.0,1.0,1.0);
		glPushMatrix();
		glTranslatef(CenterOfSimulation.x, CenterOfSimulation.y, CenterOfSimulation.z);
		renderSphere(0.02*RadiusOfLeftAtrium,20,20);
		glPopMatrix();
	}

	// Drawing other nodes
	if(Simulation.DrawNodesFlag == 1 || Simulation.DrawNodesFlag == 2)  //if we're drawing half(1) or all(2) of the nodes
	{
		for(int i = 1; i < NumberOfNodes; i++) // Start at 1 to skip the pulse node and go through all nodes
		{
			if(Simulation.DrawFrontHalfFlag == 1 || Simulation.DrawNodesFlag == 1) //if we're only drawing the front half of the nodes
			{
				if(CenterOfSimulation.z - 0.001 < Node[i].position.z)  //draw only the nodes in the front.
				{
					glColor3d(Node[i].color.x, Node[i].color.y, Node[i].color.z);
					glPushMatrix();
					glTranslatef(Node[i].position.x, Node[i].position.y, Node[i].position.z);
					renderSphere(NodeRadiusAdjustment*RadiusOfLeftAtrium,20,20);
					glPopMatrix();
				}
			}
			else //draw all nodes
			{
				glColor3d(Node[i].color.x, Node[i].color.y, Node[i].color.z);
				glPushMatrix();
				glTranslatef(Node[i].position.x, Node[i].position.y, Node[i].position.z);
				renderSphere(NodeRadiusAdjustment*RadiusOfLeftAtrium,20,20);
				glPopMatrix();
			}	
		}
	}
	else
	{
		glPointSize(5.0);
		glBegin(GL_POINTS);
		 	for(int i = 1; i < NumberOfNodes; i++)
			{
				if(Simulation.DrawFrontHalfFlag == 1)
				{
					if(CenterOfSimulation.z - 0.001 < Node[i].position.z)  // Only drawing the nodes in the front.
					{
						glColor3d(Node[i].color.x, Node[i].color.y, Node[i].color.z);
						if(Node[i].isDrawNode)
						{
							glVertex3f(Node[i].position.x, Node[i].position.y, Node[i].position.z);
						}
					}
				}
				else
				{
					glColor3d(Node[i].color.x, Node[i].color.y, Node[i].color.z);
					if(Node[i].isDrawNode)
					{
						glVertex3f(Node[i].position.x, Node[i].position.y, Node[i].position.z);
					}
				}
			}
		glEnd();
	}
	
	// Drawing muscles
	glLineWidth(LineWidth);
	for(int i = 0; i < NumberOfNodes; i++)
	{
		for(int j = 0; j < MUSCLES_PER_NODE; j++)
		{
			muscleNumber = Node[i].muscle[j];
			if(muscleNumber != -1)
			{
				k = Muscle[muscleNumber].nodeA;
				if(k == i) 
				{
					k = Muscle[muscleNumber].nodeB;
				}
				
				if(Simulation.DrawFrontHalfFlag == 1)
				{
					if(CenterOfSimulation.z - 0.001 < Node[i].position.z && CenterOfSimulation.z - 0.001 < Node[k].position.z)  // Only drawing the nodes in the front.
					{
						glColor3d(Muscle[muscleNumber].color.x, Muscle[muscleNumber].color.y, Muscle[muscleNumber].color.z);
						glBegin(GL_LINES);
							glVertex3f(Node[i].position.x, Node[i].position.y, Node[i].position.z);
							glVertex3f(Node[k].position.x, Node[k].position.y, Node[k].position.z);
						glEnd();
					}
				}
				else
				{
					glColor3d(Muscle[muscleNumber].color.x, Muscle[muscleNumber].color.y, Muscle[muscleNumber].color.z);
					glBegin(GL_LINES);
						glVertex3f(Node[i].position.x, Node[i].position.y, Node[i].position.z);
						glVertex3f(Node[k].position.x, Node[k].position.y, Node[k].position.z);
					glEnd();
				}
			}
		}	
	}
	
	// Puts a ball at the location of the mouse if a mouse function is on.
	if(Simulation.isInMouseFunctionMode)
	{
		//glColor3d(1.0, 1.0, 1.0);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    		glEnable(GL_DEPTH_TEST);
		glColor4f(1.0, 1.0, 1.0, 1.0);
		glPushMatrix();
		glTranslatef(MouseX, MouseY, MouseZ);
		
		renderSphere(HitMultiplier*RadiusOfLeftAtrium,20,20);
		//renderSphere(5.0*NodeRadiusAdjustment*RadiusOfAtria,20,20);
		glPopMatrix();
		glDisable(GL_BLEND);
	}
	
	// Saves the picture if a movie is being recorded.
	if(Simulation.isRecording)
	{
		glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, Buffer);
		fwrite(Buffer, sizeof(int)*XWindowSize*YWindowSize, 1, MovieFile);
	}

	//glfwSwapBuffers(Window);
}

/*
 This function prints all the run information to the terminal screen.
*/
void terminalPrint()
{
	//system("clear");
	//printf("\033[0;34m"); // blue.
	//printf("\033[0;36m"); // cyan
	//printf("\033[0;33m"); // yellow
	//printf("\033[0;31m"); // red
	//printf("\033[0;32m"); // green
	printf("\033[0m"); // back to white.
	
	printf("\n");
	printf("\033[0;33m");
	printf("\n **************************** Simulation Stats ****************************");
	printf("\033[0m");
	
	printf("\n Total run time = %7.2f milliseconds", RunTime);
	
	//printf("\n Driving beat node is %d.", EctopicEvents[0].node);
	printf("\n The beat rate is %f milliseconds.", Node[PulsePointNode].beatPeriod);
	printf("\n");
	
	if(Simulation.isInAdjustMuscleAreaMode || Simulation.isInAdjustMuscleLineMode) 
	{
		printf("\n Muscle refractory period multiplier =");
		printf("\033[0;36m");
		printf(" %f", RefractoryPeriodAdjustmentMultiplier);
		printf("\033[0m");
		printf("\n Muscle electrical conduction speed multiplier =");
		printf("\033[0;36m");
		printf(" %f", MuscleConductionVelocityAdjustmentMultiplier);
		printf("\033[0m");
	}
	
	for(int i = 0; i < NumberOfNodes; i++)
	{
		if(Node[i].isBeatNode && i != PulsePointNode)
		{
			printf("\n Ectopic Beat Node = %d Rate = %f milliseconds.", i, Node[i].beatPeriod);
		}
	}
	
	printf("\033[0;33m");
	printf("\n **************************** Terminal Commands ****************************");
	printf("\033[0m");
	printf("\n h: Help");
	printf("\n c: Recenter View");
	printf("\n S: Screenshot");
	printf("\n k: Save Current Run");
	printf("\n B: Lengthen Beat");
	printf("\n b: Shorten Beat");
	printf("\n ?: Find Front and Top Nodes");
	printf("\n");
	
	printf("\n Toggles");
	printf("\n r: Run/Pause            - ");
	if (Simulation.isPaused == false) 
	{
		printf("\033[0;32m");
		printf(BOLD_ON "Simulation Running" BOLD_OFF);
	} 
	else 
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Simulation Paused" BOLD_OFF);
	}
	
	printf("\n u: Contraction On/Off   - ");
	if (Simulation.ContractionisOn == true) 
	{
		printf("\033[0;32m");
		printf(BOLD_ON "Muscle Contractions on" BOLD_OFF);
	} 
	else 
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Muscle Contractions off" BOLD_OFF);
	}
	
	printf("\n g: Front/Full           - ");
	if (Simulation.DrawFrontHalfFlag == 0) printf(BOLD_ON "Full" BOLD_OFF); else printf(BOLD_ON "Front" BOLD_OFF);
	printf("\n n: Nodes Off/Half/Full  - ");
	if (Simulation.DrawNodesFlag == 0) printf(BOLD_ON "Off" BOLD_OFF); else if (Simulation.DrawNodesFlag == 1) printf(BOLD_ON "Half" BOLD_OFF); else printf(BOLD_ON "Full" BOLD_OFF);
	printf("\n v: Orthogonal/Frustum   - ");
	if (Simulation.ViewFlag == 0) printf(BOLD_ON "Orthogonal" BOLD_OFF); else printf(BOLD_ON "Frustum" BOLD_OFF);
	printf("\n m: Video On/Off         - ");
	if (!Simulation.isRecording) 
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Video Recording Off" BOLD_OFF); 
	}
	else 
	{
		printf("\033[0;32m");
		printf(BOLD_ON "Video Recording On" BOLD_OFF);
	}
	printf("\n");
	printf("\n Views" );
	
	printf("\n 7 8 9 | LL  SUP RL" );
	printf("\n 4 5 6 | PA  INF Ref" );
	printf("\n 1 2 3 | LOA AP  ROA" );
	
	printf("   You are in");
	printf("\033[0;36m");
	printf(BOLD_ON " %s", ViewName);
	printf("\033[0m" BOLD_OFF);
	printf(" view");
	
	printf("\n");
	printf("\n Adjust views");
	printf("\n w/s: CCW/CW x-axis");
	printf("\n d/a: CCW/CW y-axis");
	printf("\n z/Z: CCW/CW z-axis");
	printf("\n e/E: In/Out Zoom");
	printf("\n");
	printf("\n Set Mouse actions");
	
	printf("\n !: Ablate ---------------------- ");
	if (Simulation.isInAblateMode) 
	{
		printf("\033[0;36m");
		printf(BOLD_ON "On" BOLD_OFF); 
	}
	else printf(BOLD_ON "Off" BOLD_OFF);
	
	printf("\n @: Ectopic Beat ---------------- ");
	if (Simulation.isInEctopicBeatMode) 
	{
		printf("\033[0;36m");
		printf(BOLD_ON "On" BOLD_OFF); 
	}
	else printf(BOLD_ON "Off" BOLD_OFF);
	
	printf("\n #: Ectopic Trigger ------------- ");
	if (Simulation.isInEctopicEventMode) 
	{
		printf("\033[0;36m");
		printf(BOLD_ON "On" BOLD_OFF); 
	}
	else printf(BOLD_ON "Off" BOLD_OFF);
	
	printf("\n $: Muscle Adjustment Area Mode - ");
	if (Simulation.isInAdjustMuscleAreaMode) 
	{
		printf("\033[0;36m");
		printf(BOLD_ON "On" BOLD_OFF); 
	}
	else printf(BOLD_ON "Off" BOLD_OFF);
	
	printf("\n %%: Muscle Adjustment Line Mode - ");
	if (Simulation.isInAdjustMuscleLineMode) 
	{
		printf("\033[0;36m");
		printf(BOLD_ON "On" BOLD_OFF); 
	}
	else printf(BOLD_ON "Off" BOLD_OFF);
	
	printf("\n ^: Identify Node --------------- ");
	if (Simulation.isInFindNodeMode) 
	{
		printf("\033[0;36m");
		printf(BOLD_ON "On" BOLD_OFF); 
	}
	else printf(BOLD_ON "Off" BOLD_OFF);
	
	printf("\n ): Turns all Mouse functions off.");
	printf("\n");
	printf("\n [/]: (left/right bracket) Increase/Decrease mouse selection area.");
	printf("\n      The current selection area is");
	printf("\033[0;36m");
	printf(" %f", HitMultiplier);
	printf("\033[0m");
	printf(" times the radius of atrium.");
	printf("\033[0;33m");
	printf("\n ********************************************************************");
	printf("\033[0m");
	printf("\n");
}

/*
 This function prints the help menu to the terminal screen.
*/
void helpMenu()
{
	//system("clear");
	//Pause = 1;
	printf("\n The simulation is paused.");
	printf("\n");
	printf("\n h: Help");
	printf("\n q: Quit");
	printf("\n r: Run/Pause (Toggle)");
	printf("\n g: View front half only/View full image (Toggle)");
	printf("\n n: Nodes off/half/full (Toggle)");
	printf("\n v: Orthogonal/Frustum projection (Toggle)");
	printf("\n");
	printf("\n m: Movie on/Movie off (Toggle)");
	printf("\n S: Screenshot");
	printf("\n");
	printf("\n Views: 7 8 9 | LL  SUP RL" );
	printf("\n Views: 4 5 6 | PA  INF Ref" );
	printf("\n Views: 1 2 3 | LOA AP  ROA" );
	printf("\n");
	printf("\n c: Recenter image");
	printf("\n w: Counterclockwise rotation x-axis");
	printf("\n s: Clockwise rotation x-axis");
	printf("\n d: Counterclockwise rotation y-axis");
	printf("\n a: Clockwise rotation y-axis");
	printf("\n z: Counterclockwise rotation z-axis");
	printf("\n Z: Clockwise rotation z-axis");
	printf("\n e: Zoom in");
	printf("\n E: Zoom out");
	printf("\n");
	printf("\n [ or ]: Increases/Decrease the selection area of the mouse");
	printf("\n shift 0: Turns off all mouse action.");
	printf("\n shift 1: Turns on ablating. Left mouse ablate node. Right mouse undo ablation.");
	printf("\n shift 2: Turns on ectopic beat. Left mouse set node as an ectopic beat location.");
	printf("\n Note this action will prompt you to enter the");
	printf("\n beat period and time offset in the terminal.");
	printf("\n shift 3: Turns on one ectopic trigger.");
	printf("\n Left mouse will trigger that node to start a single pulse at that location.");
	printf("\n shift 4: Turns on muscle adjustments. Left mouse set node muscles adjustments.");
	printf("\n Note this action will prompt you to entire the ");
	printf("\n contraction, recharge, and action potential adjustment multiplier in the terminal.");
	printf("\n shift 5: Turns on find node. Left mouse displays the Id of the node in the terminal.");
	printf("\n");
	printf("\n k: Save your current muscle attributes.");
	printf("\n    (note: previous run files are ignored by git. They must be uploaded manually)");
	printf("\n ?: Find the up and front node at current view.");
	printf("\n");
}

void createGUI()
{
    // Setup ImGui window flags
    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_AlwaysAutoResize;
    
    // Main Controls Window
    ImGui::Begin("Atrium Controls", NULL, window_flags);
    
    // Run/Pause button
    if (ImGui::Button(Simulation.isPaused ? "Run" : "Pause"))
    {
        Simulation.isPaused = !Simulation.isPaused;
    }
    
    // General simulation controls
    if (ImGui::CollapsingHeader("Simulation Controls", ImGuiTreeNodeFlags_DefaultOpen))
    {
        // Contraction toggle (do we need this?? added it anyways)
        // bool contractionOn = Simulation.ContractionisOn;
        // if (ImGui::Checkbox("Contraction", &contractionOn)) 
        // {
        //     Simulation.ContractionisOn = contractionOn;
        // }
        
        // View controls
        bool frontHalf = Simulation.DrawFrontHalfFlag == 1;
        if (ImGui::Checkbox("Front Half Only", &frontHalf))
        {
            Simulation.DrawFrontHalfFlag = frontHalf ? 1 : 0;
            drawPicture();
        }
        
        // Node display options
        const char* nodeOptions[] = { "Off", "Half", "Full" };
        int nodeDisplay = Simulation.DrawNodesFlag;
        if (ImGui::Combo("Nodes Display", &nodeDisplay, nodeOptions, 3))
        {
            if (nodeDisplay != Simulation.DrawNodesFlag) // Only update if the value changes
            {
                Simulation.DrawNodesFlag = nodeDisplay;
                drawPicture();
            }
        }
        
        // Projection mode {Do we REAAALLY need this??}
        // bool frustumView = Simulation.ViewFlag == 1;
        // if (ImGui::Checkbox("Frustum View", &frustumView))
        // {
        //     if (frustumView && Simulation.ViewFlag == 0)
        //     {
        //         Simulation.ViewFlag = 1;
        //         frustumView();
        //     }
        //     else if (!frustumView && Simulation.ViewFlag == 1)
        //     {
        //         Simulation.ViewFlag = 0;
        //         orthogonalView();
        //     }
        // }
        
        // Recording
		if (Simulation.isRecording)
		{
			if (ImGui::Button("Stop Recording"))
			{
			movieOff();
			Simulation.isRecording = false;
			}
		}
		else
		{
			if (ImGui::Button("Record Video"))
			{
			movieOn();
			Simulation.isRecording = true;
			}
		}
        // Screenshot
        if (ImGui::Button("Screenshot"))
        {
            screenShot();
        }
    }
    
		// View angle controls
	if (ImGui::CollapsingHeader("View Controls", ImGuiTreeNodeFlags_DefaultOpen))
	{
		// Predefined views
		if (ImGui::Button("PA"))
		{ 
			setView(4); 
			copyNodesToGPU(); 
			drawPicture(); 
		}
		ImGui::SameLine();
		if (ImGui::Button("AP"))  
		{
			setView(2); 
			copyNodesToGPU(); 
			drawPicture(); 
		}
		ImGui::SameLine();
		if (ImGui::Button("Ref"))
		{ 
			setView(6); 
			copyNodesToGPU(); 
			drawPicture(); 
		}
		
		if (ImGui::Button("LAO"))
		{ 
			setView(1); 
			copyNodesToGPU(); 
			drawPicture(); 
		}
		ImGui::SameLine();
		if (ImGui::Button("RAO"))
		{ 
			setView(3); 
			copyNodesToGPU(); 
			drawPicture(); 
		}
		ImGui::SameLine();
		if (ImGui::Button("LL"))
		{ 
			setView(7); 
			copyNodesToGPU(); 
			drawPicture(); 
		}


		if (ImGui::Button("RL"))
		{ 
			setView(9); 
			copyNodesToGPU(); 
			drawPicture(); 
		}
		ImGUI::SameLine();
		if (ImGui::Button("SUP"))
		{ 
			setView(8); 
			copyNodesToGPU(); 
			drawPicture(); 
		}
		ImGui::SameLine();
		if (ImGui::Button("INF"))
		{ 
			setView(5); 
			copyNodesToGPU(); 
			drawPicture(); 
		}
	}
    
	// Mouse mode selection
	if (ImGui::CollapsingHeader("Mouse Functions", ImGuiTreeNodeFlags_DefaultOpen))
	{
		// Display current mouse mode
		ImGui::Text("Current Mode: ");
		if (!Simulation.isInMouseFunctionMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(1.0f, 1.0f, 1.0f, 1.0f), "Mouse Off");
		}
		else if (Simulation.isInAblateMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Ablate Mode");
		}
		else if (Simulation.isInEctopicBeatMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Ectopic Beat");
		} 
		else if (Simulation.isInEctopicEventMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.0f, 0.5f, 1.0f, 1.0f), "Ectopic Trigger");
		} 
		else if (Simulation.isInAdjustMuscleAreaMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Adjust Area");
		} 
		else if (Simulation.isInAdjustMuscleLineMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Adjust Line");
		} 
		else if (Simulation.isInFindNodeMode) 
		{
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.5f, 0.0f, 1.0f, 1.0f), "Identify Node");
		}

		// Mouse mode buttons
		if (ImGui::Button("Mouse Off"))
		{
			mouseFunctionsOff();
			Simulation.isInMouseFunctionMode = false;
		}

		if (ImGui::Button("Ablate Mode")) 
		{
			mouseAblateMode();
			Simulation.isInMouseFunctionMode = true;
			Simulation.isInAblateMode = true;
		}

		if (ImGui::Button("Ectopic Beat")) 
		{
			mouseEctopicBeatMode();
			Simulation.isInMouseFunctionMode = true;
			Simulation.isInEctopicBeatMode = true;
		}

		if (ImGui::Button("Ectopic Trigger")) 
		{
			mouseEctopicEventMode();
			Simulation.isInMouseFunctionMode = true;
			Simulation.isInEctopicEventMode = true;
		}

		if (ImGui::Button("Adjust Area"))
		{
			mouseAdjustMusclesAreaMode();
			Simulation.isInMouseFunctionMode = true;
			Simulation.isInAdjustMuscleAreaMode = true;
		}

		if (ImGui::Button("Adjust Line")) 
		{
			mouseAdjustMusclesLineMode();
			Simulation.isInMouseFunctionMode = true;
			Simulation.isInAdjustMuscleLineMode = true;
		}

		if (ImGui::Button("Identify Node")) 
		{
			mouseIdentifyNodeMode();
			Simulation.isInMouseFunctionMode = true;
			Simulation.isInFindNodeMode = true;
		}

		// Hit multiplier slider
		float hitMult = HitMultiplier;
		if (ImGui::SliderFloat("Selection Area", &hitMult, 0.0f, 0.2f, "%.3f")) 
		{
			HitMultiplier = hitMult;
			
			// Add sliders for muscle adjustment parameters when in those modes
			if (Simulation.isInAdjustMuscleAreaMode || Simulation.isInAdjustMuscleLineMode)
			{
				ImGui::Separator();
				ImGui::Text("Muscle Adjustment Parameters");
				
				float refractoryMultiplier = RefractoryPeriodAdjustmentMultiplier;
				if (ImGui::SliderFloat("Refractory Period Multiplier", &refractoryMultiplier, 0.1f, 5.0f, "%.2f")) 
				{
					RefractoryPeriodAdjustmentMultiplier = refractoryMultiplier;
				}
				ImGui::SameLine();
				if (ImGui::Button("Reset##1")) 
				{
					RefractoryPeriodAdjustmentMultiplier = 1.0f;
				}
				
				float conductionMultiplier = MuscleConductionVelocityAdjustmentMultiplier;
				if (ImGui::SliderFloat("Conduction Velocity Multiplier", &conductionMultiplier, 0.1f, 5.0f, "%.2f")) 
				{
					MuscleConductionVelocityAdjustmentMultiplier = conductionMultiplier;
				}
				ImGui::SameLine();
				if (ImGui::Button("Reset##2"))
				{
					MuscleConductionVelocityAdjustmentMultiplier = 1.0f;
				}
			}
		}
	}
    
    // Heartbeat controls
    if (ImGui::CollapsingHeader("Heartbeat Controls"))
    {
        float beatPeriod = Node[PulsePointNode].beatPeriod;
        if (ImGui::SliderFloat("Beat Period", &beatPeriod, 10.0f, 1000.0f, "%.1f ms")) 
		{
            Node[PulsePointNode].beatPeriod = beatPeriod;
            cudaMemcpy(NodeGPU, Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyHostToDevice);
            cudaErrorCheck(__FILE__, __LINE__);
        }
        
        if (ImGui::Button("+ 10ms")) 
		{
            Node[PulsePointNode].beatPeriod += 10;
            cudaMemcpy(NodeGPU, Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyHostToDevice);
            cudaErrorCheck(__FILE__, __LINE__);
        }

        ImGui::SameLine();
        if (ImGui::Button("- 10ms")) 
		{
            Node[PulsePointNode].beatPeriod -= 10;
            if(Node[PulsePointNode].beatPeriod < 0) 
			{
                Node[PulsePointNode].beatPeriod = 0;
            }
            cudaMemcpy(NodeGPU, Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyHostToDevice);
            cudaErrorCheck(__FILE__, __LINE__);
        }
        
        //Ectopic beat Sliders
        ImGui::Separator();
        ImGui::Text("Ectopic Beats");
        
        // Show sliders for each ectopic beat node
        bool hasEctopicBeats = false; // flag to see if we have any ectopic beats; consider adding to simulation struct?
        for(int i = 0; i < NumberOfNodes; i++) 
		{
            if(Node[i].isBeatNode && i != PulsePointNode) //if this is an ectopic beat node and not the "SA node"
			{
                hasEctopicBeats = true;
                
                char nodeName[32];
                sprintf(nodeName, "Ectopic Beat Node %d", i);
                
                if (ImGui::TreeNode(nodeName)) 
				{
                    float beatPeriod = Node[i].beatPeriod;
                    if (ImGui::SliderFloat("Beat Period", &beatPeriod, 10.0f, 1000.0f, "%.1f ms")) 
					{
                        Node[i].beatPeriod = beatPeriod;
                        cudaMemcpy(NodeGPU, Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyHostToDevice);
                        cudaErrorCheck(__FILE__, __LINE__);
                    }
                    
					// Set the time delay for the ectopic beat
                    float timeDelay = Node[i].beatPeriod - Node[i].beatTimer;
                    if (ImGui::SliderFloat("Time Until Next Beat", &timeDelay, 0.0f, Node[i].beatPeriod, "%.1f ms")) 
					{
                        // Convert back to beatTimer when storing
                        Node[i].beatTimer = Node[i].beatPeriod - timeDelay;
                        cudaMemcpy(NodeGPU, Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyHostToDevice);
                        cudaErrorCheck(__FILE__, __LINE__);
                    }
                    
                    //progress bar to show how far along the ectopic beat is in its cycle
                    // float phasePercent = (Node[i].beatTimer / Node[i].beatPeriod) * 100.0f;
                    // char progressLabel[32];
                    // sprintf(progressLabel, "%% of cycle: %.1f", phasePercent); //sprintf is used to format the string
                    // ImGui::ProgressBar(Node[i].beatTimer / Node[i].beatPeriod, ImVec2(-1, 0), progressLabel);
                    
					//button to remove ectopic beat nodes
                    if (ImGui::Button("Delete Ectopic Beat")) 
					{
                        Node[i].isBeatNode = false;
                        Node[i].isDrawNode = false;
                        Node[i].color = {0.0f, 1.0f, 0.0f, 1.0f}; // Reset color
                        cudaMemcpy(NodeGPU, Node, NumberOfNodes*sizeof(nodeAttributesStructure), cudaMemcpyHostToDevice);
                        cudaErrorCheck(__FILE__, __LINE__);
                    }
                    
                    ImGui::TreePop();
                }
            }
        }
        
        if (!hasEctopicBeats) //if there are no ectopic beats, show a message
		{
            ImGui::TextDisabled("No ectopic beats configured.");
            ImGui::Text("Use the Ectopic Beat button to add one.");
        }
    }
    
    // Utility functions
    if (ImGui::CollapsingHeader("Utilities"))
    {
        if (ImGui::Button("Save Settings"))
		{
            saveSettings();
        }
        
        if (ImGui::Button("Find Nodes"))
		{
            copyNodesMusclesFromGPU(); 
            float maxZ = -10000.0;
            float maxY = -10000.0;
            int indexZ = -1;
            int indexY = -1;
            
            for(int i = 0; i < NumberOfNodes; i++) 
			{
                if(maxZ < Node[i].position.z) 
				{
                    maxZ = Node[i].position.z;
                    indexZ = i;
                }
                
                if(maxY < Node[i].position.y) 
				{
                    maxY = Node[i].position.y;
                    indexY = i;
                }
            }
            
            Node[indexZ].color.x = 0.0;
            Node[indexZ].color.y = 0.0;
            Node[indexZ].color.z = 1.0;
            
            Node[indexY].color.x = 1.0;
            Node[indexY].color.y = 0.0;
            Node[indexY].color.z = 1.0;
            
            ImGui::Text("Front node index = %d", indexZ);
            ImGui::Text("Top node index = %d", indexY);
            
            drawPicture();
            copyNodesMusclesToGPU();
        }
    }
    
    ImGui::End();
    
    // Stats window
    ImGui::Begin("Simulation Stats", NULL, window_flags);
    ImGui::Text("Run time: %.2f ms", RunTime);
    ImGui::Text("Beat rate: %.2f ms", Node[PulsePointNode].beatPeriod);
    
    if(Simulation.isInAdjustMuscleAreaMode || Simulation.isInAdjustMuscleLineMode) 
	{
        ImGui::Separator();
        ImGui::Text("Refractory multiplier: %.3f", RefractoryPeriodAdjustmentMultiplier);
        ImGui::Text("Conduction multiplier: %.3f", MuscleConductionVelocityAdjustmentMultiplier);
    }
    
	// Print ectopic beat nodes and their periods
    ImGui::Separator();
    for(int i = 0; i < NumberOfNodes; i++) 
	{
        if(Node[i].isBeatNode && i != PulsePointNode) 
		{
            ImGui::Text("Ectopic Beat Node %d: %.2f ms", i, Node[i].beatPeriod);
        }
    }
  
    ImGui::End();
}