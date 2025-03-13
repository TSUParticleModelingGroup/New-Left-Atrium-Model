/*
 This file contains:
 1: All the functions that determine how to orient and view the simulation.
 2: all the functions that draw the actual simulation. 
 3: The functions that print to the linux terminal all the setting of the simulation.
 In short this file holds the functions that present information to the user.
 
 The functions are listed below in the order they appear.
 
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
void orthogonalView()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-RadiusOfLeftAtrium, RadiusOfLeftAtrium, -RadiusOfLeftAtrium, RadiusOfLeftAtrium, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	Switches.ViewFlag = 0;
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
	Switches.ViewFlag = 1;
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
	if(view == 4)
	{
		PAView();
		strcpy(ViewName, "PA");
	}
	if(view == 2)
	{
		APView();
		strcpy(ViewName, "AP");
	}
	if(view == 3)
	{
		APView();
		rotateYAxis(-PI/6.0);
		strcpy(ViewName, "RAO");
	}
	if(view == 1)
	{
		APView();
		rotateYAxis(PI/3.0);
		strcpy(ViewName, "LAO");
	}
	if(view == 7)
	{
		APView();
		rotateYAxis(PI/2.0);
		strcpy(ViewName, "LL");
	}
	if(view == 9)
	{
		APView();
		rotateYAxis(-PI/2.0);
		strcpy(ViewName, "RL");
	}
	if(view == 8)
	{
		APView();
		rotateXAxis(PI/2.0);
		strcpy(ViewName, "SUP");
	}
	if(view == 5)
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
	if(!Switches.IsPaused) glColor3d(0.0,1.0,0.0);
	else glColor3d(1.0,0.0,0.0);

	glPushMatrix();
	glTranslatef(Node[PulsePointNode].position.x, Node[PulsePointNode].position.y, Node[PulsePointNode].position.z);
	glutSolidSphere(0.03*RadiusOfLeftAtrium,20,20);
	glPopMatrix();
	
	// Drawing center node
	//This draws a center node at the center of the simulation for debugging purposes
	if(false) // false turns it off, true turns it on.
	{
		glColor3d(1.0,1.0,1.0);
		glPushMatrix();
		glTranslatef(CenterOfSimulation.x, CenterOfSimulation.y, CenterOfSimulation.z);
		glutSolidSphere(0.02*RadiusOfLeftAtrium,20,20);
		glPopMatrix();
	}

	// Drawing other nodes
	if(Switches.DrawNodesFlag == 1 || Switches.DrawNodesFlag == 2)  //if we're drawing half(1) or all(2) of the nodes
	{
		for(int i = 1; i < NumberOfNodes; i++) // Start at 1 to skip the pulse node and go through all nodes
		{
			if(Switches.DrawFrontHalfFlag == 1 || Switches.DrawNodesFlag == 1) //if we're only drawing the front half of the nodes
			{
				if(CenterOfSimulation.z - 0.001 < Node[i].position.z)  //draw only the nodes in the front.
				{
					glColor3d(Node[i].color.x, Node[i].color.y, Node[i].color.z);
					glPushMatrix();
					glTranslatef(Node[i].position.x, Node[i].position.y, Node[i].position.z);
					glutSolidSphere(NodeRadiusAdjustment*RadiusOfLeftAtrium,20,20);
					glPopMatrix();
				}
			}
			else //draw all nodes
			{
				glColor3d(Node[i].color.x, Node[i].color.y, Node[i].color.z);
				glPushMatrix();
				glTranslatef(Node[i].position.x, Node[i].position.y, Node[i].position.z);
				glutSolidSphere(NodeRadiusAdjustment*RadiusOfLeftAtrium,20,20);
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
				if(Switches.DrawFrontHalfFlag == 1)
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
				
				if(Switches.DrawFrontHalfFlag == 1)
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
	if(Switches.IsInMouseFunctionMode)
	{
		//glColor3d(1.0, 1.0, 1.0);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    		glEnable(GL_DEPTH_TEST);
		glColor4f(1.0, 1.0, 1.0, 1.0);
		glPushMatrix();
		glTranslatef(MouseX, MouseY, MouseZ);
		
		glutSolidSphere(HitMultiplier*RadiusOfLeftAtrium,20,20);
		//glutSolidSphere(5.0*NodeRadiusAdjustment*RadiusOfAtria,20,20);
		glPopMatrix();
		glDisable(GL_BLEND);
	}

	glutSwapBuffers();
	
	// Saves the picture if a movie is being recorded.
	if(Switches.MovieIsOn)
	{
		glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, Buffer);
		fwrite(Buffer, sizeof(int)*XWindowSize*YWindowSize, 1, MovieFile);
	}
}

/*
 This function prints all the run information to the terminal screen.
*/
void terminalPrint()
{
	system("clear");
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
	
	if(Switches.IsInAdjustMuscleAreaMode || Switches.IsInAdjustMuscleLineMode) 
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
	if (Switches.IsPaused == false) 
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
	if (Switches.ContractionIsOn == true) 
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
	if (Switches.DrawFrontHalfFlag == 0) printf(BOLD_ON "Full" BOLD_OFF); else printf(BOLD_ON "Front" BOLD_OFF);
	printf("\n n: Nodes Off/Half/Full  - ");
	if (Switches.DrawNodesFlag == 0) printf(BOLD_ON "Off" BOLD_OFF); else if (Switches.DrawNodesFlag == 1) printf(BOLD_ON "Half" BOLD_OFF); else printf(BOLD_ON "Full" BOLD_OFF);
	printf("\n v: Orthogonal/Frustum   - ");
	if (Switches.ViewFlag == 0) printf(BOLD_ON "Orthogonal" BOLD_OFF); else printf(BOLD_ON "Frustum" BOLD_OFF);
	printf("\n m: Video On/Off         - ");
	if (!Switches.MovieIsOn) 
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
	if (Switches.IsInAblateMode) 
	{
		printf("\033[0;36m");
		printf(BOLD_ON "On" BOLD_OFF); 
	}
	else printf(BOLD_ON "Off" BOLD_OFF);
	
	printf("\n @: Ectopic Beat ---------------- ");
	if (Switches.IsInEctopicBeatMode) 
	{
		printf("\033[0;36m");
		printf(BOLD_ON "On" BOLD_OFF); 
	}
	else printf(BOLD_ON "Off" BOLD_OFF);
	
	printf("\n #: Ectopic Trigger ------------- ");
	if (Switches.IsInEctopicEventMode) 
	{
		printf("\033[0;36m");
		printf(BOLD_ON "On" BOLD_OFF); 
	}
	else printf(BOLD_ON "Off" BOLD_OFF);
	
	printf("\n $: Muscle Adjustment Area Mode - ");
	if (Switches.IsInAdjustMuscleAreaMode) 
	{
		printf("\033[0;36m");
		printf(BOLD_ON "On" BOLD_OFF); 
	}
	else printf(BOLD_ON "Off" BOLD_OFF);
	
	printf("\n %%: Muscle Adjustment Line Mode - ");
	if (Switches.IsInAdjustMuscleLineMode) 
	{
		printf("\033[0;36m");
		printf(BOLD_ON "On" BOLD_OFF); 
	}
	else printf(BOLD_ON "Off" BOLD_OFF);
	
	printf("\n ^: Identify Node --------------- ");
	if (Switches.IsInFindNodeMode) 
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
	system("clear");
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

