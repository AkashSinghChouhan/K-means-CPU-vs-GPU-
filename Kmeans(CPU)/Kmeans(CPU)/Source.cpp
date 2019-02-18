
#include "Header.h"



float colorGrid = 0.0f;
using namespace std;
unsigned int textureId;
bool showKmeans = false;

double winW = 25.0, winH = 25.0;

const double lenghGrid = 25.0;

int widthWin = 1000, heightWin = 1000;

double total_elapsed = 0;

int n = 1000;


void display(void);
void renderPoints(void);
void initialization();
void changeViewPort(int width, int height);
void keyboard(unsigned char key, int x, int y);
void initializeData();
void mouseMovement(int x, int y);
void camera(void);




//angle of rotation
float xpos = 0, ypos = 0, zpos = 0, xrot = 0, yrot = 0, angle = 0.0;

float lastx = 500, lasty = 500;
float xrotrad, yrotrad;










DataFrame dataPoints;


int main(int argc, char* argv[]) {

	std::cout << "\nEnter number of data points and number of centroids resp. :  ";
	cin >> n >> k;
	
	initializeData();
	//std::ifstream stream("points.txt");
	//if (!stream) {
	//	std::cerr << "Could not open file: " << argv[1] << std::endl;
	//	std::exit(EXIT_FAILURE);
	//}
	//std::string line;
	//while (std::getline(stream, line)) {
	//	Point point;
	//	std::istringstream line_stream(line);
	//	//size_t label;
	//	line_stream >> point.x >> point.y >> point.z;
	//	point.label = 0;
	//	dataPoints.push_back(point);
	//}

	glutInit(&argc, argv);  // Initialize GLUT

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);   // Set up some memory buffers for our display

	glutInitWindowSize(1000, 1000);  // Set the window size

	glutInitWindowPosition(0,0);
	glutCreateWindow("Test phase");   // Create the window with the title 

	  // Bind the two functions (above) to respond when necessary

	glutDisplayFunc(display);

	glutReshapeFunc(changeViewPort);

	glutMotionFunc(mouseMovement);

	glutKeyboardFunc(keyboard);
	

	initialization();

	

	
	
	glutMainLoop();
	getchar();
	return 0;
}




void display(void)
{

	glClearColor(0.0,0.0,0.0, 0.1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	camera();
	renderPoints();

	glutSwapBuffers();
}


void renderPoints(void)
{
	if (dataPoints.size() != 0)
	{
		glPointSize(6.0f);
		glEnable(GL_POINT_SMOOTH);

		glTranslatef(0.0f, 0.0f, -2000.0f);

		if (showKmeans)
		{
			glEnable(GL_TEXTURE_1D);
			glBegin(GL_POINTS);
			for (int i = 0; i < dataPoints.size(); i++) {
				
				for (int j = 0; j < means.size(); j++) {

					
					if (dataPoints[i].label==means[j].label) { 

						glTexCoord1d((double)means[j].label/(means.size()-1));
						glVertex3f(dataPoints[i].getX(), dataPoints[i].getY(), dataPoints[i].getY());
					
					
					}

				}


			}
			glEnd();

			glDisable(GL_TEXTURE_1D);
		}

		
		


		else
		{
			glColor3f(1.0, 0.5, 0.5);
			glBegin(GL_POINTS);


			for (int i = 0; i < dataPoints.size(); i++)
			{
				glVertex3f(dataPoints[i].getX(), dataPoints[i].getY(), dataPoints[i].getY());
			}
			

			glEnd();
		}


		glDisable(GL_POINT_SMOOTH);
	}
}

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'r':
		showKmeans = false;
		break;
	case 's':
		means.clear();
		
		means.resize(k);
		total_elapsed = 0;
		for (int run = 0; run < number_of_runs; ++run) {
			const auto start = std::chrono::high_resolution_clock::now();
			means = k_means(dataPoints, k, iterations);
			const auto end = std::chrono::high_resolution_clock::now();
			const auto duration =
				std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
			total_elapsed += duration.count();
		}
		std::cout << "\n";
		std::cerr << "Execution time: " << total_elapsed / number_of_runs << "s ("
			<< number_of_runs << " runs)" << std::endl;
		
		
		for (size_t cluster = 0; cluster < k; ++cluster) {
			std::cout << "For k = " << cluster << " Centroids are " << means[cluster].getX() << " " << means[cluster].getY() << " " << means[cluster].getZ() << std::endl;
		}
		showKmeans = true;
		break;
	case 'd':
		k--;
		k = std::max(k, 2);
		break;
	case 'i':
		k++;
		break;
	default:
		break;
	}
	glutPostRedisplay();
}

void changeViewPort(int width, int height)
{
	GLfloat aspect = (GLfloat)width / (GLfloat)height;
	glViewport(0, 0, width, height);


	widthWin = width;
	heightWin = height;


	winW = (int)(width / lenghGrid + 0.5);
	winH = (int)(height / lenghGrid + 0.5);


	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	
	gluPerspective(90.0f, aspect, 0.1f, 2000.0f);


	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	



}

void initialization()
{

	glGenTextures(1, &textureId);

	glBindTexture(GL_TEXTURE_1D, textureId);

	GLfloat texture[] = { 1.0f, 0.0f, 0.0f, 1.0f,
		0.0f, 1.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 1.0f, 1.0f,
		0.8f, 0.3f, 0.8f, 0.8f,
		0.5f, 0.7f, 0.2f, 0.7f,
		0.0f, 0.0f, 6.0f, 1.0f,
		0.3f, 0.7f, 0.8f, 0.6f,
		1.0f, 1.0f, 0.0f, 0.5f,
		0.8f, 0.8f, 0.5f, 0.8f,
		0.0f, 0.3f, 8.0f, 1.0f,
		0.25f,0.25f,0.25f,1.0f,
		0.0f, 0.5f ,1.0f ,1.0f,
		0.5f,0.0f,0.75f,1.0f,
		0.3f, 0.6f ,0.7f ,1.0f,
		0.75f, 0.5f ,0.55f ,1.0f,
		0.2f, 0.9f ,0.95f ,0.6f,
		0.3f, 0.8f ,0.35f ,1.0f,
		0.2f, 0.7f ,1.0f ,0.7f,
		0.2f, 0.55f ,0.75f ,1.0f,
		1.0f, 0.2f ,0.5f ,1.0f,
		0.0f, 0.8f ,0.7f ,1.0f,
		0.1f, 0.3f ,0.7f ,1.0f,
		0.65f, 0.9f ,0.55f ,1.0f,
		0.47f, 0.2f ,0.8f ,0.9f
	};

	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 4, 0, GL_RGBA, GL_FLOAT, texture);
	
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

	
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
}

void initializeData() {
	srand(time(NULL));
	
	for (int i = 0; i < n; i++) {
		Point point;
	    point.x= rand() % 2000 - 1000;
		point.y = rand() % 2000 - 1000;
		point.z = rand() % 2000 - 1000;
		point.label = 0;
		dataPoints.push_back(point);
	
	
	}

}


void camera(void) {
	glRotatef(xrot / 15, 1.0, 0.0, 0.0);  //rotate our camera on the x - axis(left and right)
	glRotatef(yrot / 15, 0.0, 1.0, 0.0);  //rotate our camera on the y - axis(up and down)
	glTranslated(-xpos, -ypos, -zpos); //translate the screen to the position of our camera
}
void mouseMovement(int x, int y) {
	int diffx = x - lastx; //check the difference between the current x and the last x position
	int diffy = y - lasty; //check the difference between the current y and the last y position
	lastx = x; //set lastx to the current x position
	lasty = y; //set lasty to the current y position
	xrot += (float)diffy; //set the xrot to xrot with the additionof the difference in the y position
	yrot += (float)diffx;    //set the xrot to yrot with the addition of the difference in the x position


}