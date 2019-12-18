/*
 * stack.h
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

#include <stdlib.h>
#include <pthread.h>

#ifndef STACK_H
#define STACK_H

// Define a stack node struct
typedef struct Node {
  int val;
  struct Node *next;
} Node;

struct stack
{
//  int top;          // Pool stack top
//  Node pool[1000];  // Memory pool, individual per thread
  Node *head;
  #if NON_BLOCKING == 0
    pthread_mutex_t mutex;
  #endif
};
typedef struct stack stack_t;

struct poolStack  {
  Node* head[8];
  #if NON_BLOCKING == 0
    pthread_mutex_t mutex;
  #endif
};
typedef struct poolStack poolStack_t;

int stack_push(stack_t *stack, poolStack_t *pool_stack, int value, int poolIdx);
int stack_pop(stack_t *stack, poolStack_t *pool_stack, int poolIdx);
int stack_pop_ABA(stack_t* stack, poolStack_t* pool_stack, int poolIdx);
/* Use this to check if your stack is in a consistent state from time to time */
int stack_check(stack_t *stack);
#endif /* STACK_H */
