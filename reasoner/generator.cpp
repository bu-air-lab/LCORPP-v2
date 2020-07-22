#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <time.h>
//#define NUM_SAMPLE 1000

using namespace std;

class Sample{
  public:
    int weather; // sunny 0; rainy 1
    int car; // car 0; sport 1; truck 2;
    int time; // busy 0; normal 1
    int road; // crowded 0; empty 1
    int intention; // cooperative 0; not cooperative 1
};

/*
read file contents from filename and push into vector list
*/
void readfile(string filename, vector<float> &list){
  ifstream file(filename);
  if(file.is_open()){ //checking whether the file is open or not
    string line;
    while(getline(file, line)){
      if(line.size() > 0){
        list.push_back(stof(line));
      }
    }
  }else{
    cout << "Can't open Data!\n";
  }
  file.close();
}

int main(){
  vector<Sample> samples; //containts all the samples
  vector<float> wlist; //contians weather type probability
  vector<float> clist; //contians car type probability
  vector<float> tlist; //contians time type probability
  vector<float> rlist; //contians road condition probability
  vector<float> ilist; //contians intention(cooperative) probability

  //read from probability file
  readfile("weather.txt", wlist);
  readfile("car_type.txt", clist);
  readfile("time_type.txt", tlist);
  readfile("road_crowded.txt", rlist);
  readfile("cooperative.txt", ilist);

  string filename;
  int num_sample;
  cout << "Type the amount of samples: \n"; // Type a number and press enter
  cin >> num_sample; // Get user input from the keyboard
  cout << "Your number is: " << num_sample << endl; // Display the input value
  cout << "Type the name of output file: \n";
  cin >> filename;
  cout << "Output filename is: " << filename << endl;

  //generate samples according to NUM_SAMPLE and assign non-parent nodes characters
  for(int i = 0; i < num_sample; i++){
    Sample s; //create a new sample object
    float random; //store the random number we generated
    int prob; //store the probability we get from the file

    //according to the weather probability, generate weather type
    random = rand() % 10;
    prob = 0;
    for(int i = 0; i < wlist.size(); i++){
      prob += wlist[i] * 10;
      if(random < prob){
        s.weather = i;
        break;
      }
    }

    //according to the car probability, generate car type
    //srand(time(NULL));
    random = rand() % 10;
    prob = 0;
    for(int i = 0; i < clist.size(); i++){
      prob += clist[i] * 10;
      if(random < prob){
        s.car = i;
        break;
      }
    }

    //according to the time probability, generate time type
    //srand(time(NULL));
    random = rand() % 10;
    prob = 0;
    for(int i = 0; i < tlist.size(); i++){
      prob += tlist[i] * 10;
      if(random < prob){
        s.time = i;
        break;
      }
    }
    samples.push_back(s);
  }

  //according to the road_crowded probability, generate road condition type
  for(int i = 0; i < num_sample; i++){
    Sample current_sample = samples[i];
    float random; //store the random number we generated
    int prob; //store the probability we get from the file

    //srand(time(NULL));
    random = rand() % 10;
    prob = 0;
    //sunny and busy
    if(current_sample.weather == 0 && current_sample.time == 0){
      prob = rlist[0] * 10;
      if(random < prob) samples[i].road = 0;
      else samples[i].road = 1 ;
    //sunny and normal
    }else if(current_sample.weather == 0 && current_sample.time == 1){
      prob = rlist[1] * 10;
      if(random < prob) samples[i].road = 0;
      else samples[i].road =1 ;
    //rainy and busy
    }else if(current_sample.weather == 1 && current_sample.time == 0){
      prob = rlist[2] * 10;
      if(random < prob) samples[i].road = 0;
      else samples[i].road =1 ;
        //rainy and normal
    }else if(current_sample.weather == 1 && current_sample.time == 1){
      prob = rlist[3] * 10;
      if(random < prob) samples[i].road = 0;
      else samples[i].road =1 ;
    }
  }

  //according to the cooperative probability, generate cooperative characteers
  for(int i = 0; i < num_sample; i++){
    Sample current_sample = samples[i];
    float random; //store the random number we generated
    int prob; //store the probability we get from the file

    //sunny busy car crowded
    //srand(time(NULL));
    random = rand() % 10;
    prob = 0;
    if(current_sample.weather == 0 && current_sample.time == 0 && current_sample.car == 0 && current_sample.road == 0){
      prob = ilist[0] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //sunny busy car empty
    }else if(current_sample.weather == 0 && current_sample.time == 0 && current_sample.car == 0 && current_sample.road == 1){
      prob = ilist[1] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //sunny busy sport crowded
    }else if(current_sample.weather == 0 && current_sample.time == 0 && current_sample.car == 1 && current_sample.road == 0){
      prob = ilist[2] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //sunny busy sport empty
    }else if(current_sample.weather == 0 && current_sample.time == 0 && current_sample.car == 1 && current_sample.road == 1){
      prob = ilist[3] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //sunny busy truck crowded
    }else if(current_sample.weather == 0 && current_sample.time == 0 && current_sample.car == 2 && current_sample.road == 0){
      prob = ilist[4] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //sunny busy truck empty
    }else if(current_sample.weather == 0 && current_sample.time == 0 && current_sample.car == 2 && current_sample.road == 1){
      prob = ilist[5] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //sunny normal car crowded
    }else if(current_sample.weather == 0 && current_sample.time == 1 && current_sample.car == 0 && current_sample.road == 0){
      prob = ilist[6] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //sunny normal car empty
    }else if(current_sample.weather == 0 && current_sample.time == 1 && current_sample.car == 0 && current_sample.road == 1){
      prob = ilist[7] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //sunny normal sport crowded
    }else if(current_sample.weather == 0 && current_sample.time == 1 && current_sample.car == 1 && current_sample.road == 0){
      prob = ilist[8] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //sunny normal sport empty
    }else if(current_sample.weather == 0 && current_sample.time == 1 && current_sample.car == 1 && current_sample.road == 1){
      prob = ilist[9] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //sunny normal truck crowded
    }else if(current_sample.weather == 0 && current_sample.time == 1 && current_sample.car == 2 && current_sample.road == 0){
      prob = ilist[10] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //sunny normal truck empty
    }else if(current_sample.weather == 0 && current_sample.time == 1 && current_sample.car == 2 && current_sample.road == 1){
      prob = ilist[11] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //rainy busy car crowded
    }else if(current_sample.weather == 1 && current_sample.time == 0 && current_sample.car == 0 && current_sample.road == 0){
      prob = ilist[12] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //rainy busy car empty
    }else if(current_sample.weather == 1 && current_sample.time == 0 && current_sample.car == 0 && current_sample.road == 1){
      prob = ilist[13] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //rainy busy sport crowded
    }else if(current_sample.weather == 1 && current_sample.time == 0 && current_sample.car == 1 && current_sample.road == 0){
      prob = ilist[14] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //rainy busy sport empty
    }else if(current_sample.weather == 1 && current_sample.time == 0 && current_sample.car == 1 && current_sample.road == 1){
      prob = ilist[15] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //rainy busy truck crowded
    }else if(current_sample.weather == 1 && current_sample.time == 0 && current_sample.car == 2 && current_sample.road == 0){
      prob = ilist[16] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //rainy busy truck empty
    }else if(current_sample.weather == 1 && current_sample.time == 0 && current_sample.car == 2 && current_sample.road == 1){
      prob = ilist[17] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //rainy normal car crowded
    }else if(current_sample.weather == 1 && current_sample.time == 1 && current_sample.car == 0 && current_sample.road == 0){
      prob = ilist[18] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //rainy normal car empty
    }else if(current_sample.weather == 1 && current_sample.time == 1 && current_sample.car == 0 && current_sample.road == 1){
      prob = ilist[19] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //rainy normal sport crowded
    }else if(current_sample.weather == 1 && current_sample.time == 1 && current_sample.car == 1 && current_sample.road == 0){
      prob = ilist[20] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //rainy normal sport empty
    }else if(current_sample.weather == 1 && current_sample.time == 1 && current_sample.car == 1 && current_sample.road == 1){
      prob = ilist[21] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //rainy normal truck crowded
    }else if(current_sample.weather == 1 && current_sample.time == 1 && current_sample.car == 2 && current_sample.road == 0){
      prob = ilist[22] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    //rainy normal truck empty
    }else if(current_sample.weather == 1 && current_sample.time == 1 && current_sample.car == 2 && current_sample.road == 1){
      prob = ilist[23] * 10;
      if(random < prob) samples[i].intention = 0;
      else samples[i].intention = 1 ;
    }
  }

  ofstream outfile(filename);
  //according to the samples' characters, generate training Data
  for(int i = 0; i < num_sample; i++){
    Sample tmp = samples[i];

    if(tmp.weather == 0){
      outfile << "Weather(Sunny, Sample" << i << ")\n";
      outfile << "!Weather(Rainy, Sample" << i << ")\n";
    }else{
      outfile << "Weather(Rainy, Sample" << i << ")\n";
      outfile << "!Weather(Sunny, Sample" << i << ")\n";
    }

    if(tmp.time == 0){
      outfile << "Time(Busy, Sample" << i << ")\n";
      outfile << "!Time(Normal, Sample" << i << ")\n";
    }else{
      outfile << "!Time(Busy, Sample" << i << ")\n";
      outfile << "Time(Normal, Sample" << i << ")\n";
    }

    if(tmp.car == 0){
      outfile << "Vehicle(Car, Sample" << i << ")\n";
      outfile << "!Vehicle(Sport, Sample" << i << ")\n";
      outfile << "!Vehicle(Truck, Sample" << i << ")\n";
    }else if(tmp.car == 1){
      outfile << "Vehicle(Sport, Sample" << i << ")\n";
      outfile << "!Vehicle(Car, Sample" << i << ")\n";
      outfile << "!Vehicle(Truck, Sample" << i << ")\n";
    }else{
      outfile << "Vehicle(Truck, Sample" << i << ")\n";
      outfile << "!Vehicle(Car, Sample" << i << ")\n";
      outfile << "!Vehicle(Sport, Sample" << i << ")\n";
    }

    if(tmp.road == 0){
      outfile << "Road(Crowded, Sample" << i << ")\n";
      outfile << "!Road(Empty, Sample" << i << ")\n";
    }else{
      outfile << "Road(Empty, Sample" << i << ")\n";
      outfile << "!Road(Crowded, Sample" << i << ")\n";
    }

    if(tmp.intention == 0){
      outfile << "Cooperative(Sample" << i << ")\n";
    }else{
      outfile << "!Cooperative(Sample" << i << ")\n";
    }

    outfile << "\n";
  }
  outfile.close();
}
