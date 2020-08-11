/*IA_2020.c  -24/06/2020 - LOUA Pivot Remy && DIALLO Djiby Bocar : Projet Informatique en C */

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
/*definition des structures pour l'image*/
#define TAILLE 80
typedef struct
{
  int sizeImg;
  int width;
  int height;
  int classeImg;
  int* imgTab;
  int taille;
}type;
typedef struct _NODE
{
  type data;
  struct _NODE* next;
}NODE;
typedef struct
{
  int size;
  NODE* head;
  NODE* tail;
}LIST;
/*fin des structures pour l'image*/

/*definition des structures pour le reseau*/
typedef struct _NEURONE
{
  int size_weight;
  double biais;
  double* w_weight;
  double delta;
  double d_b;
  double* d_w;
}NEURONE;
typedef struct _LAYERS
{
  double* tabIn;
  double* tabOut;
  int nin;
  int nout;
  NEURONE* tab;
}LAYERS;
typedef struct _NOEUD
{
  LAYERS* layers;
  struct _NOEUD* next;
  struct _NOEUD* prev;
}NOEUD;
typedef struct _NETWORK
{
  int size;
  NOEUD* head;
  NOEUD* tail;
}NETWORK;

void loadNetwork(NETWORK* net, FILE* pfile);
void addLayerToNetwork(NETWORK* network, LAYERS* layers);
NOEUD* initNOEUD(LAYERS* layers);
NETWORK* initNetwork();
int estVideNetwork(NETWORK* network);
void VerificationReseau(NETWORK* network, char* filename);
int imgClass(double* tabOut);
double* evalNetwork(NETWORK* network, double* inputs);
double* imgNormalisation(int* tabImg, int sizeImg);
FILE* loadcsv(int argc, char* argv[]);
NODE* initNode(type data);
LIST* initListe();
void addTail(LIST* list, type data);
void affiche(LIST* list);
char* get_line(FILE* fichier);
void loadImg(FILE* file, LIST* list);
void saveBMP(int width, int height, int* data, const char* filename);
void Interface(int argc, char* argv[]);
FILE* InterLoadImage();
FILE* InterLoadNetwork();
char* InterfaceControleChaine(char* str);
void avancerGetchar();
void toUpper(char* str);

/*fonction des statistiques */
double mean(LAYERS* lay, int nbNeu);
double standardDev(LAYERS* lay, int nbNeu);
double maximum(LAYERS* lay, int nbNeu);
double minimum(LAYERS* lay, int nbNeu);
void statInfosBiais(NETWORK* net);
void statInfosPoids(NETWORK* network);
void InfosStatistiques();

/*fonction d'apprentissage */
LAYERS* initLayers(int nin, int nout);
void apprentissageNetwork(NETWORK* network, LIST* images);
void printNetwork(NETWORK* network);
void printLayers(LAYERS* layers);
void printNeurone(NEURONE* neurone);
void propagation(NETWORK* network, double* inputs);
void retropropagation(NETWORK* network, NODE* node);
void updateNetwork(NETWORK* network, double eta);
void updateDw_And_Db(NETWORK* network);


int main(int argc, char* argv[])
{
  Interface(argc, argv);
  // InfosStatistiques();

  return EXIT_SUCCESS;
}
/********************
 * @brief						cette fonction permet d'allouer la memoire pour une liste la liste doublement chaine
 *                  qui est ici dans notre cas le NETWORK
 * *****************/
NETWORK* initNetwork()
{
  NETWORK* network = (NETWORK*)malloc(sizeof(NETWORK));
  network->size = 0;
  network->head = NULL;
  network->tail = NULL;
  return network;
}
/********************
 * @brief						cette fonction permet de charger le tableau de neurone (biais + poids) a notre reseau
 * *****************/
void loadNetwork(NETWORK* net, FILE* pfile)
{
  long lsize;
  char* buffer;
  size_t result;
  if (pfile == NULL) { fputs("File error ", stderr); exit(EXIT_FAILURE); }
  /*Calcul de nombre de caractere du fichier*/
  fseek(pfile, 0, SEEK_END);
  lsize = ftell(pfile);
  rewind(pfile); /*remet le cuseur en debut du fichier*/
  /*allocation de la memoire necessaire*/
  buffer = (char*)malloc(sizeof(char) * lsize);
  /*copie du contenu du fichier dans le buffer*/
  result = fread(buffer, 1, lsize, pfile);
  if (result != lsize) { fputs("Reading error ", stderr); exit(EXIT_FAILURE); }

  int nbre_ligne = 0;
  /* On compte le nombre de ligne pour avoir une variable de controle de plus*/
  for (int i = 0; buffer[i]; i++)
  {
    if (buffer[i] == '\n')
      nbre_ligne++;
  }
  const char* delimiter = "'\n';";
  char* ch = strtok(buffer, delimiter);
  while ((ch != NULL) && (nbre_ligne > 0))
  {
    nbre_ligne--;
    int j = 0;
    int i = 0;
    LAYERS* layers = (LAYERS*)malloc(sizeof(LAYERS));
    layers->nin = atoi(ch);
    ch = strtok(NULL, delimiter);
    layers->nout = atoi(ch);
    layers->tabIn = (double*)calloc(layers->nin, sizeof(double));
    layers->tabOut = (double*)calloc(layers->nout, sizeof(double));
    layers->tab = (NEURONE*)calloc(layers->nout, sizeof(NEURONE));
    /*allocation du tableau de poids*/
    for (i = 0; i < layers->nout; i++)
    {
      layers->tab[i].w_weight = (double*)calloc(layers->nin, sizeof(double));
      layers->tab[i].d_w = (double*)calloc(layers->nin, sizeof(double));
    }
    for (i = 0; i < layers->nout; i++)
    {
      nbre_ligne--;
      ch = strtok(NULL, delimiter);
      layers->tab[i].biais = atof(ch);
      layers->tab[i].size_weight = layers->nin;
      for (j = 0; j < layers->nin; j++)
      {
        ch = strtok(NULL, delimiter);
        layers->tab[i].w_weight[j] = atof(ch);
      }
    }
    addLayerToNetwork(net, layers);/*on ajoute le layers a notre reseau*/
    ch = strtok(NULL, delimiter);
  }
  free(buffer);
}
/********************
 * @brief						cette fonction permet d'ajouter un neurone a notre reseau
 * *****************/
void addLayerToNetwork(NETWORK* network, LAYERS* layers)
{
  NOEUD* noeud = initNOEUD(layers);
  noeud->next = NULL;
  noeud->prev = network->tail;
  if (network->size > 0)
  {
    network->tail->next = noeud;
    noeud->layers->nin = noeud->prev->layers->nout;
  }
  else
    network->head = noeud;
  network->tail = noeud;
  network->size += 1;
}
/********************
 * @brief						cette fonction permet d'ajouter un layers a notre NOEUD
 * *****************/
NOEUD* initNOEUD(LAYERS* layers)
{
  NOEUD* noeud = (NOEUD*)malloc(sizeof(NOEUD));
  noeud->layers = (LAYERS*)malloc(sizeof(LAYERS));
  noeud->layers->tabIn = (double*)malloc(layers->nin * sizeof(double));
  noeud->layers->tabOut = (double*)malloc(layers->nout * sizeof(double));
  noeud->layers->tab = (NEURONE*)malloc(layers->nout * sizeof(NEURONE));
  for (int i = 0; i < layers->nout; i++)
  {
    noeud->layers->tab[i].w_weight = (double*)calloc(layers->tab[i].size_weight, sizeof(double));
    noeud->layers->tab[i].d_w = (double*)calloc(layers->tab[i].size_weight, sizeof(double));
  }
  for (int i = 0; i < layers->nout; i++)
  {
    noeud->layers->tab[i].biais = layers->tab[i].biais;
    noeud->layers->tab[i].d_b = layers->tab[i].d_b;
    noeud->layers->tab[i].delta = layers->tab[i].delta;

    noeud->layers->tab[i].size_weight = layers->tab[i].size_weight;
    for (int j = 0; j < layers->tab[i].size_weight; j++)
    {
      noeud->layers->tab[i].w_weight[j] = layers->tab[i].w_weight[j];
      noeud->layers->tab[i].d_w[j] = layers->tab[i].d_w[j];
    }
    noeud->layers->tabOut[i] = layers->tabOut[i];
  }
  for (int i = 0; i < layers->nin; i++)
    noeud->layers->tabIn[i] = layers->tabIn[i];

  noeud->layers->nin = layers->nin;
  noeud->layers->nout = layers->nout;

  noeud->next = NULL;
  noeud->prev = NULL;
  return noeud;
}
/********************
 * @brief						cette fonction permet de verifier si le reseau est vide
 * *****************/
int estVideNetwork(NETWORK* network)
{
  return (network->head == NULL);
}
/********************
 * @brief						cette fonction permet de verifier si le reseau est vide
 *                  filename : est le nom du fichier csv qui sera creer apres le passe de la fonction
 * *****************/
void VerificationReseau(NETWORK* network, char* filename)
{
  FILE* file = fopen(filename, "w");
  if (file == NULL)
  {
    fprintf(stderr, "Erreur d'ouverture de votre fichier du fichier \n");
    exit(EXIT_FAILURE);
  }
  int size_w;
  NOEUD* noeud = network->head;
  for (int i = 0; i < network->size; i++)
  {
    fprintf(file, "%d;%d\n", noeud->layers->nin, noeud->layers->nout);
    for (int j = 0; j < noeud->layers->nout; j++)
    {
      size_w = noeud->layers->tab[j].size_weight;
      fprintf(file, "%lf;", noeud->layers->tab[j].biais);
      for (int k = 0; k < size_w - 1; k++)
        fprintf(file, "%lf;", noeud->layers->tab[j].w_weight[k]);
      fprintf(file, "%lf\n", noeud->layers->tab[j].w_weight[size_w - 1]);
    }
    noeud = noeud->next;
  }
  fclose(file);
}
/********************
 * @brief						cette fonction d'evaluer le reseau avec un tableau d'image normalise avec 255 soit des
 *                   valeur entre 0 et 1
 * *****************/
double* evalNetwork(NETWORK* network, double* inputs)
{
  double v = 0.0;
  double out = 0.0;
  int i = 0;
  NOEUD* noeud = NULL;
  for (int j = 0; j < network->head->layers->nin; j++)
    network->head->layers->tabIn[j] = inputs[j];
  noeud = network->head;
  for (i = 0; i < network->size; i++)
  {
    /*pour chaque neurone on calcul la sortie*/
    for (int k = 0; k < noeud->layers->nout; k++)
    {
      v = noeud->layers->tab[k].biais;
      for (int j = 0; j < noeud->layers->nin; j++)
        v += noeud->layers->tabIn[j] * noeud->layers->tab[k].w_weight[j];
      out = 1.0 / (1.0 + exp(-v));
      noeud->layers->tabOut[k] = out;
    }
    /*transfert du buffer de sortie dans le buffer d'entree du layer suivant */
    if (noeud->next != NULL)
    {
      for (int k = 0; k < noeud->layers->nout; k++)
        noeud->next->layers->tabIn[k] = noeud->layers->tabOut[k];
    }
    if (noeud->next == NULL)
      break;
    noeud = noeud->next;
  }
  return noeud->layers->tabOut;
}
/********************
 * @brief						cette fonction permet de retrouver le chiffre correspondant a partir du tableau
 *                  sortie finale de taille 10.
 * *****************/
int imgClass(double* tabOut)
{
  int indice = 0;
  double seuil = 0.0;
  for (int i = 0; i < 10; i++)
  {
    if (tabOut[i] > seuil)
    {
      seuil = tabOut[i];
      indice = i;
    }
  }
  return indice;
}
/********************
 * @brief						cette fonction permet de permet de normaliser un tableau image. et retour un double
 *                  contenant l'image avec des valeurs entre 0 et 1
 * *****************/
double* imgNormalisation(int* tabImg, int sizeImg)
{
  double* tab = (double*)calloc(sizeImg, sizeof(double));
  for (int i = 0; i < sizeImg; i++)
    tab[i] = tabImg[i] / 255.0;
  return tab;
}
/********************
 * @brief						cette fonction permet de charger le network a partir de la ligne de commande avec la
 *                  forme suivante : .\ia.exe load network_30_10.csv
 * *****************/
FILE* loadcsv(int argc, char* argv[])
{
  FILE* file = NULL;
  printf("le nombre de argc : %d", argc);
  if ((argc == 4) && (!strcmp(argv[1], "load")))
  {
    if (!strcmp(argv[2], "network"))
    {
      file = fopen(argv[3], "r");
      if (file == NULL)
      {
        printf("Erreur d'ouveture du fichier : %s\n", argv[3]);
        exit(EXIT_FAILURE);
      }
    }
    else if (!strcmp(argv[2], "img"))
    {
      file = fopen(argv[3], "r");
      if (file == NULL)
      {
        printf("Erreur d'ouveture du fichier : %s\n", argv[3]);
        exit(EXIT_FAILURE);
      }
    }
    else
    {
      printf("Erreur-->argv[2] : n'est pas conforme.\nargv[2] = network, pour charger reseau\nargv[2] = img ,pour charger une image de test\n");
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    printf("Merci de verifier les arguments de ia.exe \n");
    exit(EXIT_FAILURE);
  }
  return file;
}

/*fin de code seance 4*/

/********************
 * @brief						cette fonction permet d'initialiser un noeud avec une donne "data"
 * *****************/
NODE* initNode(type data)
{
  NODE* node = malloc(sizeof(NODE));
  node->data.imgTab = malloc(data.taille * sizeof(int));
  if (node == NULL)
  {
    fprintf(stderr, "Erreur : initNode ");
    exit(EXIT_FAILURE);
  }
  node->data.classeImg = data.classeImg;
  node->data.height = data.height;
  node->data.width = data.width;
  node->data.taille = data.taille;
  node->data.sizeImg = data.sizeImg;
  for (int i = 0; i < data.taille; i++)
    node->data.imgTab[i] = data.imgTab[i];
  return node;
}
/********************
 * @brief						cette fonction permet d'allouer la memoire pour une liste chainee simple
 * *****************/
LIST* initListe()
{
  LIST* list = malloc(sizeof(LIST));
  if (list == NULL)
  {
    fprintf(stderr, "Erreur : initListe() \n");
    exit(EXIT_FAILURE);
  }
  list->size = 0;
  list->head = NULL;
  list->tail = NULL;
  return list;
}
/********************
 * @brief						cette fonction permet d'ajouter une "data" a la liste chainee simple
 * *****************/
void addTail(LIST* list, type data)
{
  NODE* node = initNode(data);
  node->next = NULL;
  if (list->size > 0)
    list->tail->next = node;
  else
    list->head = node;
  list->tail = node;
  list->size += 1;
}
/********************
 * @brief						cette fonction permet d'afficher imgTab de la liste chaine simple
 * *****************/
void affiche(LIST* list)
{
  NODE* node = list->head;
  while (node != NULL)
  {
    for (int i = 0; i < node->data.taille; i++)
      printf("%d ", node->data.imgTab[i]);
    printf("\n\n\n");
    node = node->next;
  }
}
char* get_line(FILE* fichier)
{
  size_t size = TAILLE;
  char* line = (char*)malloc(size);
  // Si l'allocation de la memoire se passe bien
  if (line != NULL) {
    int c;
    size_t i = 0;
    // s'il n'y a pas de caracteres dans le fichier, on arrete tout*/
    if ((c = fgetc(fichier)) == EOF) {
      free(line);
      line = NULL;
      //fclose(fichier);
    }
    // Sinon s'il y a des caracteres dans le fichier on le lit */
    else {
      do {

        // On Aloue de l'espace memoire si necessaire
        if (i == size - 1) {
          char* tmp = realloc(line, size + TAILLE);
          /* s'il y a eu erreur d'allocation de memoire,
  on libere l'espace que l'on avait alloue,et ferme le fichier et on retourne NULL */
          if (tmp != NULL) {
            line = tmp;
            size += TAILLE;
          }
          else {
            free(line);
            line = NULL;
            //fclose(fichier);
          }
        }
        line[i] = c;
        i++;
      } while ((c = fgetc(fichier)) != '\n' && c != EOF);
    }

    /* SI 'il y a eu une erreur de lecture ,
    on libere l'espace allou�, on vide le buffer et on retourne NULL */

    if (ferror(fichier) != 0) {
      free(line);
      line = NULL;
      //fclose(fichier);
    }
    /*SINON on rajoute le \0 terminal*/
    if (line != NULL) {
      line[i] = '\0';
    }
  }
  return line;
}
/********************
 * @brief						ecriture d'une image en .bmp
 * *****************/
void saveBMP(int width, int height, int* data, const char* filename)
{
  int pad = (4 - (3 * width) % 4) % 4;
  int filesize = 54 + (3 * width + pad) * height;
  unsigned char header[54] = { 'B','M', 0,0,0,0, 0,0,0,0, 54,0,0,0,
40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,24,0 };
  char padding[3] = { 0,0,0 };
  unsigned char* img;
  FILE* file;
  int  i;
  for (i = 0; i < 4; i++)
  {
    header[2 + i] = (unsigned char)((filesize >> (8 * i)) & 255);
    header[18 + i] = (unsigned char)((width >> (8 * i)) & 255);
    header[22 + i] = (unsigned char)((height >> (8 * i)) & 255);
  }
  img = malloc(sizeof(char) * 3 * (size_t)width * (size_t)height);
  if (img == NULL)
    printf("Error could not allocate memory !\n");
  else
  {
    for (i = 0; i < width * height; i++)
    {
      int color = data[i];
      img[3 * i + 2] = img[3 * i + 1] = img[3 * i] = (unsigned char)(color &
        0xFF);
    }
    file = fopen(filename, "wb");
    if (file == NULL)
      printf("Error: could not open file '%s' !\n", filename);
    else
    {
      fwrite(header, 54, 1, file);
      for (i = height - 1; i >= 0; i--)
      {
        fwrite((char*)(img + 3 * width * i), 3 * (size_t)width, 1, file);
        fwrite(padding, (unsigned int)pad, 1, file);
      }
      fclose(file);
    }
    free(img);
  }
}
/********************
 * @brief						cette fonction permet de charger une image dans la liste
 * *****************/
void loadImg(FILE* file, LIST* list)
{
  char* chaine = get_line(file);
  size_t t = strlen(chaine);
  type* data = (type*)malloc(sizeof(type));
  data->imgTab = (int*)malloc(sizeof(int));
  char* ch = strtok(chaine, ";");
  int sizeImg = atoi(ch);
  ch = strtok(NULL, ";");
  int height = atoi(ch);
  ch = strtok(NULL, ";");
  int width = atoi(ch);
  while ((chaine = get_line(file)) != NULL)
  {
    data->sizeImg = sizeImg;
    data->height = height;
    data->width = width;
    int a = 0;
    double n = 0;
    int m = 0;
    ch = strtok(chaine, ";");
    data->classeImg = atoi(ch);
    ch = strtok(NULL, ";");
    do
    {
      if (strchr(ch, ','))
      {
        ch[1] = '.';
        t = strlen(ch);
        n = atof(ch);
        for (int i = 0; i < t - 2; i++)
          n = n * 10;
        m = (int)n;
        for (int i = 0; i < m; i++)
        {
          *(data->imgTab + a) = 0;
          data->imgTab = (int*)realloc(data->imgTab, (a + 1) * sizeof(type));
          a++;
        }
      }
      else
      {
        n = atoi(ch);
        *(data->imgTab + a) = n;
        data->imgTab = (int*)realloc(data->imgTab, (a + 1) * sizeof(type));
        a++;
      }
      ch = strtok(NULL, ";");
    } while (ch != NULL);
    data->taille = a;
    addTail(list, *data);
  }
  free(data->imgTab);
  free(data);
  free(ch);
  free(chaine);
}


void Interface(int argc, char* argv[])
{
  FILE* file = NULL;
  FILE* pfile = NULL;
  {
    if ((argc == 4) && (!strcmp(argv[1], "load")))
    {
      if (!strcmp(argv[2], "network"))
      {
        file = fopen(argv[3], "r");
        if (file == NULL)
        {
          printf("Erreur d'ouveture du fichier : %s\n", argv[3]);
          exit(EXIT_FAILURE);
        }
      }
      else if (!strcmp(argv[2], "img"))
      {
        file = fopen(argv[3], "r");
        if (file == NULL)
        {
          printf("Erreur d'ouveture du fichier : %s\n", argv[3]);
          exit(EXIT_FAILURE);
        }
      }
      else
      {
        printf("Erreur-->argv[2] : %s n'est pas conforme.\nargv[2] = network, pour charger reseau\nargv[2] = img ,pour charger une image de test\n", argv[2]);
        exit(EXIT_FAILURE);
      }
    }
    else
    {
      printf("Merci de verifier les arguments de ia.exe \n");
      exit(EXIT_FAILURE);
    }
  }
  /*une fois ici tous un fichier au moins est ouvert*/
  NETWORK* network = initNetwork();
  LIST* images = initListe();
  if (!strcmp(argv[2], "network"))
  {
    printf("Fichier %s Ouvert , en cours de chargement...\n", argv[3]);
    loadNetwork(network, file);
    VerificationReseau(network, "verifie.csv");
    printf("fin de chargement\n");
    avancerGetchar();
    pfile = InterLoadImage();
    printf("Fichier d'image Ouvert, en cours de chargement...\n");
    loadImg(pfile, images);
    printf("fin de chargement\n");
  }
  else if (!strcmp(argv[2], "img"))
  {
    printf("Fichier %s Ouvert , en cours de chargement...\n", argv[3]);
    loadImg(file, images);
    printf("fin de chargement de %s \n", argv[3]);
    avancerGetchar();
    pfile = InterLoadNetwork();
    printf("Fichier de network Ouvert, en cours de chargement...\n");
    loadNetwork(network, pfile);
    printf("fin de chargement\n");
  }
  printf("vous avez charger le reseau et une base d'images pour le avaluer le reseau\n");
  double* tab = (double*)malloc(10 * sizeof(double));
  double* inputs = NULL;
  double  taux = 0.0;
  double  taux_erreur = 0.0;
  int  reussite = 0;
  int k = 0;
  char* ch = (char*)calloc(6, sizeof(char));
  int arret = 0;
  char* ch2 = (char*)malloc(BUFSIZ * sizeof(char));
  char* chi = (char*)malloc(BUFSIZ * sizeof(char));
  NODE* node = images->head;
  printf("format d'affichage pendant l'evaluation \n");
  printf("Numero image ==> prediction: classimg : statut : taux de reussite : taux erreur : nbre ok : nbre mauvaise prediction");
  while (node != NULL)
  {
    getchar();
    strcpy(ch, "non");
    inputs = imgNormalisation(node->data.imgTab, node->data.taille);
    tab = evalNetwork(network, inputs);
    k++;
    printf("\nImage %-6d: ", k);
    printf("==> %d\t", imgClass(tab));
    printf("%d\t", node->data.classeImg);
    if (node->data.classeImg == imgClass(tab))
    {
      reussite++;
      strcpy(ch, "ok");
    }
    printf("%s\t", ch);
    taux = ((double)reussite / k) * 100;
    printf("\t%-7g%%", taux);
    taux_erreur = 100.0 - taux;
    printf("\t%-7g%% ", taux_erreur);
    printf("\t %d", reussite);
    printf("\t %d", (k - reussite));
    node = node->next;
    arret++;
    if (arret == 100)
    {
      printf("\n\nVoulez vous continuer le pas a pas (OUI/NON)? :  NON = different de OUI \n");
      rewind(stdin);
      fgets(chi, BUFSIZ, stdin);
      rewind(stdin);
      ch2 = InterfaceControleChaine(chi);
      toUpper(ch2);
      if (!strcmp(ch2, "OUI"))
        arret = 0;
      else
        break;
    }
  }
  while (node != NULL)
  {
    strcpy(ch, "non");
    inputs = imgNormalisation(node->data.imgTab, node->data.taille);
    tab = evalNetwork(network, inputs);
    k++;
    printf("\nImage %-6d: ", k);
    printf("==> %d\t", imgClass(tab));
    printf("%d\t", node->data.classeImg);
    if (node->data.classeImg == imgClass(tab))
    {
      reussite++;
      strcpy(ch, "ok");
    }
    printf("%s\t", ch);
    taux = ((double)reussite / k) * 100;
    printf("\t%-7g%%", taux);
    taux_erreur = 100.0 - taux;
    printf("\t%-7g%% ", taux_erreur);
    printf("\t %d", reussite);
    printf("\t %d", (k - reussite));
    node = node->next;
  }
  printf("\n");
  printf("=========Apres Evaluation de %d images Nous avons ===========\n", k);
  printf("Taux d'erreur                               : %g%%\n", taux_erreur);
  printf("Taux de reussite                            : %g%%\n", taux);
  printf("Nombre d'image avec une bonne prediction    : %d\n", reussite);
  printf("Nombre d'image avec une mauvaise prediction : %d\n", (k - reussite));

  printf("************Fin Pour le Reseau  ==> Statistique !*****************");
  printf("\nVoulez vous continuer sur la statistique des couches du reseau (OUI/NON)? : NON = different de OUI \n");
  rewind(stdin);
  fgets(chi, BUFSIZ, stdin);
  rewind(stdin);
  ch2 = InterfaceControleChaine(chi);
  toUpper(ch2);
  if (!strcmp(ch2, "OUI"))
  {
    InfosStatistiques();
  }
}
FILE* InterLoadImage()
{
  FILE* file = NULL;
  char* chi = (char*)malloc(sizeof(char) * BUFSIZ);
  char* ch2 = (char*)malloc(sizeof(char) * BUFSIZ);
  char* chaine = NULL;
  char* ch = NULL;
  printf("\nVoulez vous charger une image (OUI/NON)? :  NON = different de OUI \n");
  fgets(chi, BUFSIZ, stdin);
  ch = InterfaceControleChaine(chi);
  toUpper(ch);
  if (!strcmp(ch, "OUI"))
  {
    printf("Entrer le nom du fichier a charger \n");
    fgets(ch2, BUFSIZ, stdin);
    chaine = InterfaceControleChaine(ch2);
    file = fopen(chaine, "r");
    if (file == NULL)
    {
      fprintf(stderr, "Erreur : erreur d'ouverture du fichier ");
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    printf("Vous avez choisie de ne pas charger une image donc nous allons utiliser notre image\n");
    file = fopen("images_data.csv", "r");
    if (file == NULL)
    {
      fprintf(stderr, "Erreur : erreur d'ouverture du fichier image dans InterloadImage");
      exit(EXIT_FAILURE);
    }
  }
  return file;
}
void avancerGetchar()
{
  printf("continuer ?\n Oui = La touche entree\t Non = Crtl+c\n");
  getchar();
}
FILE* InterLoadNetwork()
{
  FILE* file = NULL;
  char ch;
  char* chi = (char*)malloc(sizeof(char) * BUFSIZ);
  char* ch2 = (char*)malloc(sizeof(char) * BUFSIZ);
  char* chaine = NULL;
  printf("\nVoulez vous charger le Network (OUI/NON)? \n");
  printf("1 = OUI et 0 = NON\n");
  fgets(chi, BUFSIZ, stdin);
  ch = atoi(chi) + 48;
  switch (ch)
  {
  case '1':
    printf("Entrer le nom du fichier network \n");
    fgets(ch2, BUFSIZ, stdin);
    chaine = InterfaceControleChaine(ch2);
    file = fopen(chaine, "r");
    if (file == NULL)
    {
      fprintf(stderr, "Erreur : erreur d'ouverture du fichier network,merci de verifier l'emplacement du fichier ");
      exit(EXIT_FAILURE);
    }
    printf("Chargement de %s en cours \n", chaine);
    break;
  default:
    printf("Vous avez choisie de ne pas charger le network donc nous allons utiliser notre network");
    file = fopen("network_100_30_10.csv", "r");
    if (file == NULL)
    {
      fprintf(stderr, "Erreur : erreur d'ouverture du fichier network,merci de verifier l'emplacement du fichier ");
      exit(EXIT_FAILURE);
    }
    break;
  }
  return file;
}
char* InterfaceControleChaine(char* str)
{
  char* chaine = NULL;
  int nbre = 0;
  int k = 0;
  for (int i = 0; str[i]; i++)
  {
    if ((str[i] != ' ') && (str[i] != '\t') && (str[i] != '\n'))
      nbre++;
  }
  chaine = (char*)malloc((nbre) * sizeof(char));
  for (int i = 0; str[i]; i++)
  {
    if ((str[i] != ' ') && (str[i] != '\t') && (str[i] != '\n'))
    {
      chaine[k] = str[i];
      k++;
    }
  }
  chaine[k] = '\0';
  return chaine;
}
void toUpper(char* str)
{
  int i;
  for (i = 0; str[i]; i++)
  {
    if (str[i] >= 'a' && str[i] <= 'z')
      str[i] = str[i] - 'a' + 'A'; /* str[i]-'a' : pour recuperer lecart entre a et str[i]*/
                                  /* et on trouve l'equivalent majuscule en fesant ecart+'A'*/
  }
}

/*fonction statistique */
// calculer la moyenne des biais
double mean(LAYERS* lay, int nbNeu)
{
  double val_moyen = 0.0;
  for (int i = 0; i < nbNeu; i++)
  {
    val_moyen += lay->tab[i].biais;;
  }
  val_moyen = val_moyen / nbNeu;
  return val_moyen;
}
// calculer la deviation standard des biais 
double standardDev(LAYERS* lay, int nbNeu)
{
  double st_devia = 0;
  for (int i = 0; i < nbNeu; i++)
  {
    st_devia += ((lay->tab[i].biais - mean(lay, nbNeu)) * (lay->tab[i].biais - mean(lay, nbNeu)));
  }

  st_devia = sqrt(st_devia / (nbNeu));

  return st_devia;
}
// Determiner le maximum des biais
double maximum(LAYERS* lay, int nbNeu)
{
  double max = lay->tab[0].biais;
  for (int i = 1; i < nbNeu; i++)
  {
    if (max < lay->tab[i].biais)
      max = lay->tab[i].biais;
  }

  return max;
}
// Determiner le min des biais
double minimum(LAYERS* lay, int nbNeu)
{
  double min = lay->tab[0].biais;
  for (int i = 0; i < nbNeu; i++)
  {
    if (lay->tab[i].biais < min)   			 // on verifie si l'element suivant est plus petit que min
      min = lay->tab[i].biais;
  }

  return min;
}
void statInfosBiais(NETWORK* network)
{
  double meanV, stdDev, min, max;

  int i = 0;
  NOEUD* noeud = NULL;

  noeud = network->head;
  printf("\t--- Donnees pour les biais ---\n");
  for (i = 0; i < network->size; i++)
  {
    printf("-Couche [%d] \n", (i + 1));

    printf(" Nombre de Neurones : %d \n", noeud->layers->nout);
    meanV = mean(noeud->layers, noeud->layers->nout);
    printf(" Valeur moyenne : %g\n", meanV);

    stdDev = standardDev(noeud->layers, noeud->layers->nout);
    printf(" Valeur de la deviation standard : %g\n", stdDev);

    max = maximum(noeud->layers, noeud->layers->nout);
    printf(" Valeur max :  %g\n", max);

    min = minimum(noeud->layers, noeud->layers->nout);
    printf(" Valeur min : %g\n", min);

    printf("\n");

    if (noeud->next == NULL) break;
    noeud = noeud->next;
  }


}

// Fonctions pour afficher les donnees statistiques pour les poids  

void statInfosPoids(NETWORK* network)
{
  double min, max;

  int i = 0;
  NOEUD* noeud = NULL;

  noeud = network->head;

  printf("\t--- Donnees pour les poids ---\n");

  for (i = 0; i < network->size; i++)
  {
    double* w_max = (double*)malloc(noeud->layers->nout * sizeof(double));
    double* w_min = (double*)malloc(noeud->layers->nout * sizeof(double));

    printf("-Couche [%d] \n", (i + 1));

    printf(" Nombre de Neurones : %5d \n", noeud->layers->nout);

    double val_moyen = 0.0;
    double poidstotal = 0.0;
    int nbTotalPoidNeurons = 0; // nombre de poids total pour tous les neurones

    for (int l = 0; l < noeud->layers->nout; l++)
    {
      // Pour chaque neurone, on addition les differents poids qui lui contient
      for (int j = 0; j < noeud->layers->tab[i].size_weight; j++) {
        poidstotal += noeud->layers->tab[l].w_weight[j];

        nbTotalPoidNeurons += noeud->layers->tab[l].size_weight;
      }
      // on a la somme des poids pour un neurone donné qu'on additionnera avec les poids suivants
      val_moyen += poidstotal;

    }

    val_moyen = val_moyen / (nbTotalPoidNeurons);


    printf(" Valeur moyenne : %g\n", val_moyen);

    double st_devia = 0;
    for (int m = 0; m < noeud->layers->nout; m++)
    {
      for (int j = 0; j < noeud->layers->tab[m].size_weight; j++) {
        st_devia += ((noeud->layers->tab[m].w_weight[j] - val_moyen) * (noeud->layers->tab[m].w_weight[j] - val_moyen));
      }
    }

    st_devia = sqrt(st_devia / nbTotalPoidNeurons);
    printf(" Valeur de la deviation standard : %g\n", st_devia);

    // Trouver maximum,  en stockant le max de chaque poids d'un neurone dans le tableau w_max et ensuite trouver le max de ce tableau

    for (int m = 0; m < noeud->layers->nout; m++) {

      double wmax = noeud->layers->tab[m].w_weight[0];
      for (int n = 1; n < noeud->layers->tab[m].size_weight; n++)
      {
        if (wmax < noeud->layers->tab[m].w_weight[n])	 wmax = noeud->layers->tab[m].w_weight[n];
      }
      w_max[m] = wmax;
    }

    max = w_max[0];
    for (int k = 1; k < noeud->layers->nout; k++)
    {
      if (max < w_max[k])	 max = w_max[k];
    }
    printf(" Valeur max :  %g\n", max);


    for (int m = 0; m < noeud->layers->nout; m++) {

      double wmin = noeud->layers->tab[m].w_weight[0];
      for (int n = 1; n < noeud->layers->tab[m].size_weight; n++)
      {
        if (wmin > noeud->layers->tab[m].w_weight[n])	 wmin = noeud->layers->tab[m].w_weight[n];
      }
      w_min[m] = wmin;
    }

    min = w_min[0];
    for (int k = 1; k < noeud->layers->nout; k++)
    {
      if (min > w_min[k])	 min = w_min[k];
    }
    printf(" Valeur min :  %g\n", min);


    printf("\n");

    if (noeud->next == NULL) break;
    noeud = noeud->next;
  }


}


void InfosStatistiques()
{
  char csvFichier[256], * p;
  FILE* fp;
  char choix[2];

  printf("Voudriez-vous charger un autre fichier Reseaux ou utiliser notre fichier csv.\nTaper sur [O/o] pour oui ou autre pour non ? ");
  fgets(choix, 2, stdin);
  int c = 0;
  while (c != '\n' && c != EOF)
  {
    c = getchar();

  }

  if (choix[0] == 'O' || choix[0] == 'o')
  {
    printf("Veuiller saisir le nom du fichier : ");
    fgets(csvFichier, 256, stdin);

    if ((p = strchr(csvFichier, '\n')) != NULL)
    {
      *p = '\0'; /* retirer la fin de ligne (retour a la ligne)*/
    }

    fp = fopen(csvFichier, "r"); // mode lecture 
    if (fp == NULL)
    {
      perror("[-] Erreur pendant l'ouverture du fichier.\n");
      exit(EXIT_FAILURE);

    }

    NETWORK* net = initNetwork();
    loadNetwork(net, fp);
    printf("**********************************************************\n");
    printf("\tRESEAU AVEC %d COUCHES\n", net->size);
    printf("**********************************************************\n");
    statInfosBiais(net);
    statInfosPoids(net);
    exit(EXIT_SUCCESS);

  }
  else {
    FILE* f = fopen("network_30_10.csv", "r");
    NETWORK* net = initNetwork();
    loadNetwork(net, f);
    printf("**********************************************************\n");
    printf("\tRESEAU AVEC %d COUCHES\n", net->size);
    printf("**********************************************************\n");
    statInfosBiais(net);
    printf("Appuyez sur entrer pour voir la suite \n");
    getchar();
    statInfosPoids(net);
  }
}


/*fonction pour l'apprentissage */
void propagation(NETWORK* network, double* inputs)
{
  double v = 0.0;
  double out = 0.0;
  int i = 0;
  NOEUD* noeud = NULL;
  for (int j = 0; j < network->head->layers->nin; j++)
    network->head->layers->tabIn[j] = inputs[j];
  noeud = network->head;
  for (i = 0; i < network->size; i++)
  {
    /*pour chaque neurone on calcul la sortie*/
    for (int k = 0; k < noeud->layers->nout; k++)
    {
      v = noeud->layers->tab[k].biais;
      for (int j = 0; j < noeud->layers->nin; j++)
        v += noeud->layers->tabIn[j] * noeud->layers->tab[k].w_weight[j];
      out = 1.0 / (1.0 + exp(-v));
      noeud->layers->tabOut[k] = out;
    }
    /*transfert du buffer de sortie dans le buffer d'entree du layer suivant */
    if (noeud->next != NULL)
    {
      for (int k = 0; k < noeud->layers->nout; k++)
        noeud->next->layers->tabIn[k] = noeud->layers->tabOut[k];
    }
    if (noeud->next == NULL)
      break;
    noeud = noeud->next;
  }
}
void apprentissageNetwork(NETWORK* network, LIST* images)
{
  double* inputs = (double*)calloc(images->head->data.taille, sizeof(double));
  NODE* node = images->head;
  NOEUD* noeud = network->head;
  while (node != NULL)
  {
    for (int i = 0; i < images->head->data.taille; i++)
      inputs[i] = node->data.imgTab[i] / 255.0;
    propagation(network, inputs);
    retropropagation(network, node);
    updateDw_And_Db(network);
    updateNetwork(network, 0.01);
    /*on remet les delta a zeros*/
    while (noeud != NULL)
    {
      for (int n = 0; n < noeud->layers->nout; n++)
      {
        noeud->layers->tab[n].delta = 0;
        noeud->layers->tab[n].d_b = 0;
        for (int m = 0; m < noeud->layers->tab[n].size_weight; m++)
          noeud->layers->tab[n].d_w[m] = 0;
      }
      noeud = noeud->next;
    }
    //VerificationReseau(network, "pendant.csv");
    node = node->next;
  }
}
void updateNetwork(NETWORK* network, double eta)
{
  NOEUD* noeud = network->head;
  long double intermediaire = 0.0;
  /*pour chaque layers*/
  while (noeud != NULL)
  {
    /*pour chaque neurone*/
    for (int i = 0; i < noeud->layers->nout; i++)
    {
      noeud->layers->tab[i].biais -= eta * noeud->layers->tab[i].d_b;
      // printf("%g\n", noeud->layers->tab[i].d_b);
      for (int j = 0; j < noeud->layers->tab[i].size_weight; j++)
      {
        //printf("avant : %lf\t ", noeud->layers->tab[i].w_weight[j]);
        intermediaire = (long double)eta * noeud->layers->tab[i].d_w[j];
        //printf("intermediaire : %lf\t ", intermediaire);
        noeud->layers->tab[i].w_weight[j] = noeud->layers->tab[i].w_weight[j] - intermediaire;
        //printf("dw      : %lf  w_apres : %lf \n", noeud->layers->tab[i].d_w[j], noeud->layers->tab[i].w_weight[j]);
      }
    }
    noeud = noeud->next;
  }
}
void updateDw_And_Db(NETWORK* network)
{
  NOEUD* noeud = network->head;
  /*pour chaque layers*/
  while (noeud != NULL)
  {
    for (int i = 0; i < noeud->layers->nout; i++)
    {
      /*pour chaque neurone*/
      noeud->layers->tab[i].d_b += noeud->layers->tab[i].delta;
      /*pour chaque d_w[i] de chaque neurone*/
      for (int j = 0; j < noeud->layers->tab[i].size_weight; j++)
      {
        noeud->layers->tab[i].d_w[j] += noeud->layers->tab[i].delta * noeud->layers->tabIn[j];
      }
    }
    noeud = noeud->next;
  }
}
void retropropagation(NETWORK* network, NODE* node)
{
  NOEUD* noeud = network->tail;
  LAYERS* layersActuel = noeud->layers; /*correspond au layer qu'on manipule a l'instant present*/
  LAYERS* layersSuivant = NULL; /* le layers suivant de layersActuel*/
  double z = 0.0;
  /*pour chaque neurone du dernier layers*/
  for (int i = 0; i < layersActuel->nout; i++)
    layersActuel->tab[i].delta = (layersActuel->tabOut[i] - (node->data.classeImg == i)) * layersActuel->tabOut[i] * (1 - layersActuel->tabOut[i]);
  do
  {
    noeud = noeud->prev;
    layersActuel = noeud->layers;
    layersSuivant = noeud->next->layers;
    /*pour chaque neurone du layers n du layer actuel*/
    for (int n = 0; n < layersActuel->nout; n++)
    {
      z = 0.0;
      /*pour chaque neurone m du layers suivant du layers actuel */
      for (int m = 0; m < layersSuivant->nout; m++)
        z += layersSuivant->tab[m].delta * layersSuivant->tab[m].w_weight[n];
      layersActuel->tab[n].delta = z * layersActuel->tabOut[n] * (1 - layersActuel->tabOut[n]);

      //printf("\n%g   %g   %g", layersActuel->tab[n].delta, layersActuel->tabOut[n],network->head->layers->tabOut[n]);
    }
  } while (noeud->prev != NULL);
}
LAYERS* initLayers(int nin, int nout)
{
  LAYERS* layers = (LAYERS*)calloc(1, sizeof(LAYERS));
  layers->nin = nin;
  layers->nout = nout;
  layers->tab = (NEURONE*)calloc(nout, sizeof(NEURONE));
  layers->tabIn = (double*)calloc(nin, sizeof(double));
  layers->tabOut = (double*)calloc(nout, sizeof(double));
  for (int i = 0; i < nout; i++)
  {
    layers->tab[i].d_w = (double*)calloc(nin, sizeof(double));
    layers->tab[i].w_weight = (double*)calloc(nin, sizeof(double));
    layers->tab[i].size_weight = nin;
  }
  return layers;
}
void printNetwork(NETWORK* network)
{
  if (estVideNetwork(network))
    printf("le reseau est vide !!\n");
  else
  {
    NOEUD* noeud = network->head;
    while (noeud != NULL)
    {
      printLayers(noeud->layers);
      noeud = noeud->next;
    }
    printf("\n");
  }
}
void printNeurone(NEURONE* neurone)
{
  int i = 0;
  printf("%g ;", neurone->biais);
  for (i = 0; i < neurone->size_weight - 1; i++)
    printf("%g ;", neurone->w_weight[i]);
  if (neurone->size_weight >= 1)
    printf("%g \n", neurone->w_weight[neurone->size_weight - 1]);
}
void printLayers(LAYERS* layers)
{
  int i = 0;
  printf("Layers : Inputs \n");
  for (i = 0; i < layers->nin; i++)
    printf("%g ", layers->tabIn[i]);
  printf("\n");
  for (i = 0; i < layers->nout; i++)
  {
    printf("neurone %d \n", i);
    printNeurone(&layers->tab[i]);
  }
  printf("\n");
  printf("Layers : Outputs \n");
  for (i = 0; i < layers->nout; i++)
    printf("%g ", layers->tabOut[i]);
  printf("\n");
}