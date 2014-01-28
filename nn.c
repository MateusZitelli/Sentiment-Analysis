#include <stdio.h>
#include <math.h>


int ni = 5, nh, no = 1, trainSetSize, trainSteps;
float desired_error;
float sigmoid(float z){
    return 1.0 / (1.0 + exp(-z));
}

float dsigmoid(float z){
    return z * (1 - z);
}

float linear(float z){
    return z;
}

float dlinear(float z){
    return 1.0;
}

float dtanh(float z){
    return 1.0 - z * z;
}

float dotproduct(float * w, float * x, int n){
    float sum = 0;
    int i;
    for(i = 0; i < n; i++){
        sum += w[i] * x[i];
    }
    return sum;
}

float runmlp(float * x, float ** wi, float ** wh, int ni, int nh, int no, float * f, float * fo){
    int i;
    for(i = 0; i < nh; i++){
        f[i] = 0;
    }
    for(i = 0; i < no; i++){
        fo[i] = 0;
    }
    for(i = 0; i < nh; i++){
        f[i] = sigmoid(dotproduct(wi[i], x, ni + 1));
    }
    for(i = 0; i < no; i++){
        fo[i] = sigmoid(dotproduct(wh[i], f, nh));
    }
}

void backpropagate(float * y, float * f, float * fo, float ** wi, float ** wh, int nh, int no, float * deltao, float * deltah){
    float * whj = (float *) malloc(no * sizeof(float));
    float sdeltao;
    int i, j;
    for(i = 0; i < no; i++){
        deltao[i] = 0;
    }
    for(i = 0; i < nh; i++){
        deltah[i] = 0;
    }
    for(i = 0; i < no; i++){
        deltao[i] = (fo[i] - y[i]) * dlinear(fo[i]);
    }
    for(i = 0; i < nh; i++){
        for(j = 0; j < no; j++){
            whj[j] = wh[j][i];
        }
        sdeltao = dotproduct(deltao, whj, no);
        deltah[i] = sdeltao * dsigmoid(f[i]);
    }
    free(whj);
}

void learn(float ** wi, float ** wh, float alpha, float * deltah, float * deltao, float * f, float * fo, float * x, int nh, int no, int ni){
    int j, i;
    for(j = 0; j < no; j++){
        for(i = 0; i < nh; i++){
            wh[j][i] -= alpha * deltao[j] * f[i];
        }
    }
    for(j = 0; j < nh; j++){
        for(i = 0; i < ni + 1; i++){
            wi[j][i] -= alpha * deltah[j] * x[i];
        }
    }
}

void mlp(float ** wi, float ** wh, int ni, int nh, int no, float alpha, float ** X, float ** Y, int s){
    float error;
    float * f = (float *) malloc(nh * sizeof(float *));
    float * fo = (float *) malloc(no * sizeof(float *));
    float * deltah = (float *) malloc(nh * sizeof(float *));
    float * deltao = (float *) malloc(no * sizeof(float *));
    int i, j, k;
    for(i = 0; i < nh; i++){
        wi[i] = (float *) malloc((ni + 1) * sizeof(float));
        for(j = 0; j < ni + 1; j++){
            wi[i][j] = rand() % 1000000 / 1000000.0;
        }
    }
    for(i = 0; i < no; i++){
        wh[i] = (float *) malloc((nh) * sizeof(float));
        for(j = 0; j < nh; j++){
            wh[i][j] = rand() % 1000000 / 1000000.0;
        }
    }
    for(i = 0; i < trainSteps; i++){
        printf("Processo:%.01f\%\n", i / (float)trainSteps * 100);
        error = 0;
        for(j = 0; j < s; j++){
            runmlp(X[j], wi, wh, ni, nh, no, f, fo);
            for(k = 0; k < no; k++){
                error += sqrt((Y[j][k] - fo[k]) * (Y[j][k] - fo[k]));
            }
            backpropagate(Y[j], f, fo, wi, wh, nh, no, deltao, deltah);
            learn(wi, wh, alpha, deltah, deltao, f, fo, X[j], nh, no, ni);
        }
        error /= s * no;
        printf("ERROR: %f STEP: %i of %i\n", error, i, trainSteps);
        if(error <= desired_error)
            break;
        if(i % 20 == 19){
            FILE *results; 
            results = fopen("r.txt","w");
            fprintf(results,"%s","[");
            for(i = 0; i < nh; i++){
                fprintf(results,"%s","[");
                for(j = 0; j < ni; j++){
                    fprintf(results,"%f,",wi[i][j]);
                }
                fprintf(results,"%s","],");
            }
            fprintf(results,"%s","]\n");
            fprintf(results,"%s","[");
            for(i = 0; i < no; i++){
                fprintf(results,"%s","[");
                for(j = 0; j < nh; j++){
                    fprintf(results,"%f,",wh[i][j]);
                }
                fprintf(results,"%s","],");
            }
            fprintf(results,"%s","]\n");
            fclose(results);
        }
    }
    free(f);
    free(fo);
    free(deltao);
    free(deltah);
}

void get_values(char * line, int stringSize, float * list, int quant, int offset){
    int i, j, sig = 1;
    char * nbuffer = (char *) malloc(100);
    int pos = 0;
    int nsize = 0;
    int numsFounded = offset;
    int intDigits = 0;
    for(i = 0; i < stringSize; i++){
        if(line[i] == ' ' || line[i] == '\n' || line[i] == EOF){
            list[numsFounded] = 0;
            if(nsize > 0){
                for(j = 0; j < nsize; j++){
                    list[numsFounded] += pow(10, nsize - j - 1) * (nbuffer[j] - 48);
                }
                list[numsFounded] *= sig;
                if(intDigits != 0){
                    list[numsFounded] /= pow(10,  nsize- intDigits);
                }

                numsFounded++;
            }
            sig = 1;
            pos = 0;
            nsize = 0;
        }else{
            if(line[i] == '.'){
                intDigits = nsize;
                continue;
            }else if(line[i] == '-'){
                sig = -1;
                continue;
            }
            nbuffer[pos++] = line[i];
            nsize++;
        }
    }
    free(nbuffer);
}

void read_trainSet(char * filename, float ** X, float ** Y,  int params){
    int size = 1024, pos, i;
    int c;
    char *buffer = (char *)malloc(size);
    int line = 0;
    FILE *f = fopen(filename, "r");
    float * n1_list = (float *) malloc(ni * sizeof(float));
    float * n2_list = (float *) malloc(no * sizeof(float));
    if(f) {
      do {
        pos = 0;
        do{
          c = fgetc(f);
          if(c != EOF) buffer[pos++] = (char)c;
          if(pos >= size - 1) {
            size *=2;
            buffer = (char*)realloc(buffer, size);
          }
        }while(c != EOF && c != '\n');
        buffer[pos] = 0;
        if(pos == 0)
            continue;
        if(line == 0 && params){
            get_values(buffer, pos, n1_list, 6, 0);
            ni = n1_list[0];
            nh = n1_list[1];
            no = n1_list[2];
            trainSetSize = n1_list[3];
            trainSteps = n1_list[4];
            desired_error = n1_list[5];
            return;
        }else if(line % 2 == 1){
            X[(line - 1) / 2][0] = 1;
            get_values(buffer, pos + 1, X[(line - 1) / 2], ni, 1);
            // for(i = 0; i < ni; i++){
            //     printf("%f ", X[(line - 1) / 2][i + 1]);
            // }
        }else if(line > 0){
            get_values(buffer, pos + 1, Y[(line - 2) / 2], no, 0);       
            // for(i = 0; i < no; i++){
            //     printf("%f ", Y[(line - 2) / 2][i]);
            // }
        }
        //printf("\n");
        ++line;
      } while(c != EOF); 
      fclose(f);           
    }
    free(buffer);
}

int main(int argc,char *argv[]){
    int i, j;
    printf("Lendo trainSet\n");
    read_trainSet("./train.txt", -1, -1, 1);
    float ** X = (float **) malloc(trainSetSize * sizeof(float *));
    float ** Y = (float **) malloc(trainSetSize * sizeof(float *));
    for(i = 0; i < trainSetSize; i++){
        X[i] = (float *) malloc(ni * sizeof(float));
        Y[i] = (float *) malloc(no * sizeof(float));
    }
    read_trainSet("./train.txt", X, Y, 0);
    printf("Leitura terminada\n");
    float ** wi = (float **) malloc(nh * sizeof(float *));
    float ** wh = (float **) malloc(no * sizeof(float *));
    float * f = (float *) malloc(nh * sizeof(float *));
    float * fo = (float *) malloc(no * sizeof(float *));
    mlp(wi, wh, ni, nh, no, 0.3, X, Y, trainSetSize);
    FILE *results; 
    results = fopen("r.txt","w");
    fprintf(results,"%s","[");
    for(i = 0; i < nh; i++){
        fprintf(results,"%s","[");
        for(j = 0; j < ni; j++){
            fprintf(results,"%f,",wi[i][j]);
        }
        fprintf(results,"%s","],");
    }
    fprintf(results,"%s","]\n");
    fprintf(results,"%s","[");
    for(i = 0; i < no; i++){
        fprintf(results,"%s","[");
        for(j = 0; j < nh; j++){
            fprintf(results,"%f,",wh[i][j]);
        }
        fprintf(results,"%s","],");
    }
    fprintf(results,"%s","]\n");
    fclose(results);
    /*for(i = 0; i < trainSetSize; i++){
        runmlp(X[i], wi, wh, ni, nh, no, f, fo);
        printf("INPUT:[");
        for(j = 0; j < ni; j++){
            printf("%f, ", X[i][j + 1]);
        }
        printf("]\nOUTPUT:[");
        for(j = 0; j < no; j++){
            printf("%f->%f, ", fo[j], Y[i][j]);
        }
        printf("]\n");
    }*/
    
    return 0;
}