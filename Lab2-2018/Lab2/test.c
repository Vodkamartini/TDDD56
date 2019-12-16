/*
 * test.c
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 *
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stddef.h>
#include <unistd.h>

#include "test.h"
#include "stack.h"
#include "non_blocking.h"

#define test_run(test)\
  printf("[%s:%s:%i] Running test '%s'... ", __FILE__, __FUNCTION__, __LINE__, #test);\
  test_setup();\
  if(test())\
  {\
    printf("passed\n\n");\
  }\
  else\
  {\
    printf("failed\n\n");\
  }\
  test_teardown();

/* Helper function for measurement */
double timediff(struct timespec *begin, struct timespec *end)
{
	double sec = 0.0, nsec = 0.0;
   if ((end->tv_nsec - begin->tv_nsec) < 0)
   {
      sec  = (double)(end->tv_sec  - begin->tv_sec  - 1);
      nsec = (double)(end->tv_nsec - begin->tv_nsec + 1000000000);
   } else
   {
      sec  = (double)(end->tv_sec  - begin->tv_sec );
      nsec = (double)(end->tv_nsec - begin->tv_nsec);
   }
   return sec + nsec / 1E9;
}

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5

#ifndef NDEBUG
int
assert_fun(int expr, const char *str, const char *file, const char* function, size_t line)
{
	if(!(expr))
	{
		fprintf(stderr, "[%s:%s:%zu][ERROR] Assertion failure: %s\n", file, function, line, str);
		abort();
		// If some hack disables abort above
		return 0;
	}
	else
		return 1;
}
#endif

stack_t *stack;
data_t data;
poolStack_t *pool_stack;
poolStack_t *pool_stack2;

#if MEASURE != 0
struct stack_measure_arg
{
  int id;
};
typedef struct stack_measure_arg stack_measure_arg_t;

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;


#if MEASURE == 1
void*
stack_measure_pop(void* arg)
  {
    stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
    int i;

    clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
    for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
      {
        stack_pop(stack, &pool_stack[args->id]);
        // See how fast your implementation can pop MAX_PUSH_POP elements in parallel
      }
    clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

    return NULL;
  }
#elif MEASURE == 2
void*
stack_measure_push(void* arg)
{

  stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
  int i;

  clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
  for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
  {
      stack_push(stack, &pool_stack[args->id], DATA_VALUE);
      // See how fast your implementation can push MAX_PUSH_POP elements in parallel
  }

  clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

  return NULL;
}
#elif MEASURE == 3
void*
stack_measure_push_pop(void* arg)
{

  stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
  int i;

  clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
  for(i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
  {
    stack_push(stack, &pool_stack[args->id], DATA_VALUE);
  }
  for(i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
  {
    stack_pop(stack, &pool_stack[args->id]);
  }
  clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

  return NULL;
}
#endif
#endif

void pop_measure_init()
{
    // Initialize your test batch
    data = DATA_VALUE;
    // Allocate a new stack and reset its values
    stack = malloc(sizeof(stack_t));
    pool_stack = malloc(sizeof(poolStack_t)*NB_THREADS);

    stack->head = malloc(sizeof(Node));
    Node* curr = stack->head;
    curr->val = DATA_VALUE;

    for( int i=0; i<MAX_PUSH_POP; ++i){
        curr->next = malloc(sizeof(Node));
        curr = curr->next;
        curr->val = DATA_VALUE;
    }
}

void push_measure_init()
{
  // Initialize your test batch
  data = DATA_VALUE;
  // Allocate a new stack and reset its values
  stack = malloc(sizeof(stack_t));
  pool_stack = malloc(sizeof(poolStack_t)*NB_THREADS);

  for( int t=0; t<NB_THREADS; ++t){
      pool_stack[t].head = malloc(sizeof(Node));
      Node* curr = pool_stack[t].head;

      for( int i=0; i<MAX_PUSH_POP/NB_THREADS; ++i){
          curr->next = malloc(sizeof(Node));
          curr = curr->next;
          curr->val = 0;
      }
  }
}


/* A bunch of optional (but useful if implemented) unit tests for your stack */
void
print_stack()
{
  Node* node = stack->head;
  printf("\n");
  while(node != NULL) {
    printf("%i ", node->val);
    node = node->next;
  }
  printf("\n");
}


void
test_init()
{
  // Initialize your test batch
  data = DATA_VALUE;

  // Allocate memory for a new stack and pool (and set all values to 0)
  stack = malloc(sizeof(stack_t));
  pool_stack = malloc(sizeof(poolStack_t));
  pool_stack2 = malloc(sizeof(poolStack_t));


// HÄR ÄR DET FEL
  pool_stack->head = malloc(sizeof(Node));
  Node* node = pool_stack->head;

  for(int i = 0; i < MAX_PUSH_POP; i++) {
    node->next = malloc(sizeof(Node));
    node = node->next;
    node->val = 0;
  }
}

void
test_setup()
{
  print_stack(pool_stack);
  // Allocate and initialize your test stack before each test
  stack_push(stack, pool_stack, 3);
  stack_push(stack, pool_stack, 2);
  stack_push(stack, pool_stack, 1);
  /*stack_push(stack, pool_stack, 1);
  stack_push(stack, pool_stack, 2);
  stack_push(stack, pool_stack, 3);
  stack_push(stack, pool_stack, 4);
  stack_push(stack, pool_stack, 5);
  stack_push(stack, pool_stack, 6);
  stack_push(stack, pool_stack, 7);
  stack_push(stack, pool_stack, 8);
  stack_push(stack, pool_stack, 9);
  stack_push(stack, pool_stack, 10);*/
}

void
test_teardown()
{
  // Do not forget to free your stacks after each test
  // to avoid memory leaks
  //free(stack); // since we do not use malloc, we cannot use free anymore, we put the "freed" elements back in the pool

  // Instead we loop through and pop the entire stack
  while(stack->head != NULL) {
    stack_pop(stack, pool_stack);
  }
}

void
test_finalize()
{
  // Destroy properly your test batch
  // I.e. now we actually want to DELETE the memory we have allocated in test_init()

  Node* node = stack->head;
  // Again, we loop through the stack like in test_init(), but we free allocated elements
  Node* temp;
  while(node != NULL) {
    temp = node;
    node = node->next;
    free(temp);
  }

  // Same for pool since we want to DELETE that one too
  node = pool_stack->head;
  while(node != NULL) {
    temp = node;
    node = node->next;
    free(temp);
  }

  // Lastly, delete the stacks
  free(stack);
  free(pool_stack);

}

int
test_push_safe()
{
  // Make sure your stack remains in a good state with expected content when
  // several threads push concurrently to it

  // Do some work
  printf("Starting push test, expecting 1 2 3, we have:");
  print_stack(stack);
  stack_push(stack, pool_stack, DATA_VALUE);
  printf("After push we expect 5 1 2 3, we got:");
  print_stack(stack);

  // check if the stack is in a consistent state
  //int res = assert(stack_check(stack));

  // check other properties expected after a push operation
  // (this is to be updated as your stack design progresses)
  // Now, the test succeeds


  return assert(stack->head->val == DATA_VALUE);
  /*res &&*/
}

int
test_pop_safe()
{
  // Same as the test above for parallel pop operation

  // For now, this test always fails
  /*int test = stack->head->next->val;
  printf("\nHead should after pop be: %i", test);
  stack_pop(stack, pool_stack);
  print_stack(stack);

  int res = assert(stack_check(stack));
  return res && assert(stack->head->val == test);*/
  printf("Starting pop test, expecting 1 2 3, we have:");
  print_stack(stack);
  Node* pool_stack_head_rem = stack->head;
  if(pool_stack_head_rem == NULL){
    return assert(stack->head == NULL);
  }

  stack_pop(stack,pool_stack);
  printf("After pop we expect 2 3, we got:");
  print_stack(stack);

  return assert(stack->head != pool_stack_head_rem);

}

// 3 Threads should be enough to raise and detect the ABA problem
#define ABA_NB_THREADS 3

// Create the ABA problem according to Lession 1
void* thread0 (void* arg) {
  printf("Start Thread 0\n Stack:");
  print_stack(stack);
  printf("Thread 0 ABA pop stack \n");
  stack_pop_ABA(stack, pool_stack);
  printf("Thread 0 ended. \n");
  return NULL;
}
void* thread1(void* arg) {
  printf("Start Thread 1\n Stack:");
  print_stack(stack);
  printf("Thread 1 pop stack two times \n");
  stack_pop(stack, pool_stack);
  stack_pop(stack, pool_stack);
  printf("Thread 1 push stack \n");
  stack_push(stack, pool_stack, 1);
  printf("Thread 1 ended. \n");
  return NULL;
}
/*
void* thread2(void* arg) {
  printf("Start Thread 2\n Stack:");
  print_stack(stack);
  printf("Stack 2 pop stack \n");
  stack_pop(stack, pool_stack);
  printf("Thread 2 ended. \n");
  return NULL;
}
*/
int
test_aba()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  int success, aba_detected = 0;
  // Write here a test for the ABA problem
  //    Stack starts out as head -> 1 -> 2 -> 3
  //    Thread 0 starts pop(), but is put on hold forcefully, is looking at 1 and next = 2
  //    Thread 1 runs pop(), the stack is now head -> 2 -> 3
  //    Thread 1 runs pop(), the stack is now head -> 3
  //    Thread 1 pushes 1 back to the stack, the stack is now head -> 1 -> 3
  //    Thread 0 is now allowed to run now and compare 1 == 1, which will pass as correct, however next is still 2 which is wrong

  // Create the two threads defined above
  pthread_t thread[2];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  pthread_create(&thread[0], &attr, thread0, NULL);
  sleep(0.2);
  pthread_create(&thread[1], &attr, thread1, NULL);
 // sleep(0.2);
 // pthread_create(&thread[2], &attr, thread2, NULL);

  pthread_join(thread[0], NULL);
  pthread_join(thread[1], NULL);
//  pthread_join(thread[2], NULL);
  printf("Stack after ABA");
  print_stack(stack);

  success = aba_detected;
  //return success;
  return assert(stack->head->next->val == 3);
#else
  // No ABA is possible with lock-based synchronization. Let the test succeed only
  return 1;
#endif
}

// We test here the CAS function
struct thread_test_cas_args
{
  int id;
  size_t* counter;
  pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void*
thread_test_cas(void* arg)
{
#if NON_BLOCKING != 0
  thread_test_cas_args_t *args = (thread_test_cas_args_t*) arg;
  int i;
  size_t old, local;

  for (i = 0; i < MAX_PUSH_POP; i++)
    {
      do {
        old = *args->counter;
        local = old + 1;
#if NON_BLOCKING == 1
      } while (cas(args->counter, old, local) != old);
#elif NON_BLOCKING == 2
      } while (software_cas(args->counter, old, local, args->lock) != old);
#endif
    }
#endif

  return NULL;
}

// Make sure Compare-and-swap works as expected
int
test_cas()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_cas_args_t args[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  size_t counter;

  int i, success;

  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  for (i = 0; i < NB_THREADS; i++)
    {
      args[i].id = i;
      args[i].counter = &counter;
      args[i].lock = &lock;
      pthread_create(&thread[i], &attr, &thread_test_cas, (void*) &args[i]);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  success = assert(counter == (size_t)(NB_THREADS * MAX_PUSH_POP));

  if (!success)
    {
      printf("Got %ti, expected %i. ", counter, NB_THREADS * MAX_PUSH_POP);
    }

  return success;
#else
  return 1;
#endif
}

int
main(int argc, char **argv)
{
setbuf(stdout, NULL);
// MEASURE == 0 -> run unit tests
#if MEASURE == 0
  test_init();

  //test_run(test_cas);
  test_run(test_push_safe);
  test_run(test_pop_safe);
  test_run(test_aba);

  test_finalize();
#else
  int i;
  pthread_t thread[NB_THREADS];
  pthread_attr_t attr;
  stack_measure_arg_t arg[NB_THREADS];
  pthread_attr_init(&attr);
#if MEASURE==1
  pop_measure_init();
#elif MEASURE==2 || MEASURE == 3
  push_measure_init();
#endif
  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < NB_THREADS; i++)
    {
      arg[i].id = i;
#if MEASURE == 1
      pthread_create(&thread[i], &attr, stack_measure_pop, (void*)&arg[i]);
#elif MEASURE == 2
      pthread_create(&thread[i], &attr, stack_measure_push, (void*)&arg[i]);
#elif MEASURE == 3
      pthread_create(&thread[i], &attr, stack_measure_push_pop, (void*)&arg[i]);
#endif
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }
  clock_gettime(CLOCK_MONOTONIC, &stop);

  // Print out results
  for (i = 0; i < NB_THREADS; i++)
    {
        printf("Thread %d time: %f\n", i, timediff(&t_start[i], &t_stop[i]));
    }
#endif

  return 0;
}
