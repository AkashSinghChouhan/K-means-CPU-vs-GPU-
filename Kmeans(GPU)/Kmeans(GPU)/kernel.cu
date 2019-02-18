



#include "header.cuh"

std::vector<float> h_x;
std::vector<float> h_y;
std::vector<float> h_z;
std::vector<int> h_label;
std::vector<float> mean_x;
std::vector<float> mean_y;
std::vector<float> mean_z;
std::vector<int> mean_label;
int number_of_elements ;


int n = 100000;

int* d_counts;

const int threads = 1024;
int blocks;

float colorGrid = 0.0f;
using namespace std;
unsigned int textureId;

bool showKmeans = false;

double winW = 25.0, winH = 25.0;

const double lenghGrid = 25.0;

int widthWin = 600, heightWin = 600;

double total_elapsed = 0;




void display(void);
void renderPoints(void);
void initialization();
void changeViewPort(int width, int height);
void keyboard(unsigned char key, int x, int y);
void initializeData();
void initializeMean();
void camera(void);
void mouseMovement(int x, int y);



//angle of rotation
float xpos = 0, ypos = 0, zpos = 0, xrot = 0, yrot = 0, angle = 0.0;

float lastx=500, lasty=500;
float xrotrad, yrotrad;





int main(int argc, char* argv[]) {

	std::cout << "\nEnter number of data points and number of centroids resp. :  ";
	cin >> n >> k;
	initializeData();
	
	
	

	glutInit(&argc, argv);  // Initialize GLUT

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH |GLUT_WINDOW_CURSOR);   // Set up some memory buffers for our display

	glutInitWindowSize(1000, 1000);  // Set the window size

	glutInitWindowPosition(0, 0);
	glutCreateWindow("Test phase");   // Create the window with the title 

	
	
	glutDisplayFunc(display);

	glutReshapeFunc(changeViewPort);
	//glutIdleFunc(display);
	glutMotionFunc(mouseMovement);
	glutKeyboardFunc(keyboard);
	
	glEnable(GL_DEPTH_TEST);
	


	initialization();





	glutMainLoop();
	getchar();
	return 0;
}




void display(void)
{

	glClearColor(0.1,0.1,0.1, 0.1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	

	camera();
	
	renderPoints();

	glutSwapBuffers();
	angle++;
}


void renderPoints(void)
{
	if (h_x.size() != 0)
	{
		glPointSize(6.0f);
		glEnable(GL_POINT_SMOOTH);

		glTranslatef(0.0f, 0.0f, -2000.0f);

		if (showKmeans)
		{
			
			glEnable(GL_TEXTURE_1D);
			glBegin(GL_POINTS);
			for (int i = 0; i < h_x.size();i++) {

				for (int j = 0; j < mean_label.size(); j++) {


					if (h_label.at(i) == mean_label.at(j)) {

						glTexCoord1d((double)mean_label.at(j) / (mean_label.size() - 1));
						glVertex3f(h_x.at(i),h_y.at(i),h_z.at(i));


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


			for (int i = 0; i < h_x.size(); i++)
			{
				glVertex3f(h_x.at(i), h_y.at(i), h_z.at(i));
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
	case 's': {

		initializeMean();

		Data d_data(number_of_elements, h_x, h_y, h_z, h_label);
		
		Data d_means(k, mean_x, mean_y, mean_z, mean_label);

		Data d_sums(k);


		cudaMalloc(&d_counts, k * sizeof(int));
		cudaMemset(d_counts, 0, k * sizeof(int));


		blocks = (number_of_elements + threads - 1) / threads;

		const auto start = std::chrono::high_resolution_clock::now();
		for (size_t iteration = 0; iteration < number_of_iterations; ++iteration) {
			cudaMemset(d_counts, 0, k * sizeof(int));
			d_sums.clear();

			assign_clusters << <blocks, threads >> > (d_data.x,
				d_data.y,
				d_data.z,
				d_data.size,
				d_means.x,
				d_means.y,
				d_means.z,
				d_sums.x,
				d_sums.y,
				d_sums.z,
				k,
				d_counts
				);
			cudaDeviceSynchronize();

			compute_new_means << <1, k >> > (d_means.x,
				d_means.y,
				d_means.z,
				d_sums.x,
				d_sums.y,
				d_sums.z,
				d_counts);
			cudaDeviceSynchronize();




		}
		const auto end = std::chrono::high_resolution_clock::now();
		const auto duration =
			std::chrono::duration_cast<std::chrono::duration<float>>(end - start);
		std::cerr << "Execution time: " << duration.count() << "s" << std::endl;


		cudaFree(d_counts);


		mean_x.resize(k, 0); mean_y.resize(k, 0); mean_z.resize(k, 0);
		cudaMemcpy(mean_x.data(), d_means.x, d_means.bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(mean_y.data(), d_means.y, d_means.bytes, cudaMemcpyDeviceToHost);
		cudaMemcpy(mean_z.data(), d_means.z, d_means.bytes, cudaMemcpyDeviceToHost);

		assign_label(mean_x,
			mean_y,
			mean_z,
			mean_label,
			h_x,
			h_y,
			h_z,
			h_label,
			k,
			number_of_elements);

		
		std::cout << "\n";
		for (size_t cluster = 0; cluster < k; ++cluster) {
			std::cout <<"For k = " <<cluster<<" Centroids are "<<mean_x[cluster] << " " << mean_y[cluster] << " "<<mean_z[cluster]<<std::endl;
		}
		showKmeans = true;
		break;
		
	}
		
	case 'l':
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
	/*std::ifstream stream();
	std::string line;
	while (std::getline(stream, line)) {
		std::istringstream line_stream(line);
		float x, y, z;

		line_stream >> x >> y >> z;
		h_x.push_back(x);
		h_y.push_back(y);
		h_z.push_back(z);
		h_label.push_back(0);
	}
	*/
	for (int i = 0; i < n; i++) {
		
		h_x.push_back(rand() % 2000 - 1000); 
		h_y.push_back(rand() % 2000 - 1000);
		h_z.push_back(rand() % 2000 - 1000);
		h_label.push_back(0);
		


	}
	number_of_elements = h_x.size();

}

void initializeMean() {

	mean_label.clear();
	
	mean_label.resize(k);
	mean_x.clear();
	mean_y.clear();
	mean_z.clear();
	mean_x.resize(k);
	mean_y.resize(k);
	mean_z.resize(k);

	srand(time(NULL));
	
	for (int i = 0; i < k; i++) {

		mean_x.push_back(rand() % 2000 - 1000);
		mean_y.push_back(rand() % 2000 - 1000);
		mean_z.push_back(rand() % 2000 - 1000);
		mean_label.push_back(i);



	}
	

}
void camera(void) {
	glRotatef(xrot/15, 1.0, 0.0, 0.0);  //rotate our camera on the x - axis(left and right)
	glRotatef(yrot/15, 0.0, 1.0, 0.0);  //rotate our camera on the y - axis(up and down)
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





