/*
 * stack.c
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
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif

int
stack_check(stack_t *stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
	// Use assert() to check if your stack is in a state that makes sense
	// This test should always pass 
	assert(1 == 1);

	// This test fails if the task is not allocated or if the allocation failed
	assert(stack != NULL);
#endif
	// The stack is always fine
	return 1;
}

int /* Return the type you prefer */
stack_push(stack_t *stack, int value)
{
#if NON_BLOCKING == 0

  pthread_mutex_lock(&stack->mutex);
  Node_t *new_node;
  new_node = malloc(sizeof(Node_t));  // Allocate memory for the new_node node
  
  new_node->val = value;
  new_node->next = stack->head;
  stack->head = new_node;

  free(new_node); // We are now done with the temp node so we free it
  pthread_mutex_unlock(&stack->mutex);

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack

  size_t casRes;

  do {

    // Do some speculative work
    Node_t *new_node;
    new_node = malloc(sizeof(Node_t));

    new_node->val = value;
    new_node->next = stack->head;
    stack->head = new_node;

    free(new_node);

    // Try Compare & Swap
    casRes = cas(
      (size_t*)&stack->head,  // Memory location / Check location
      (size_t)stack->head,    // Expected value / Old value
      (size_t)new_node        // New value
    );

  } while(casRes != stack->head);

#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check((stack_t*)1);

  return 0;
}

int /* Return the type you prefer */
stack_pop(stack_t *stack)
{
  // If the stack is empty there is nothing to pop...
  if(stack->head == NULL) {
    return -1;
  }

#if NON_BLOCKING == 0
  
  pthread_mutex_lock(&stack->mutex);

  Node_t *next_node = NULL;

  next_node = stack->head->next;
  stack->head = next_node;

  free(next_node);

  pthread_mutex_unlock(&stack->mutex);

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack
  size_t casRes;
  do {

    // Do some speculative work
    Node_t *next_node = NULL;

    next_node = stack->head->next;
    stack->head = next_node;

    free(next_node);

    // Try Compare & Swap
    casRes = cas(
      (size_t*)&stack->head,    // Memory location / Check location
      (size_t)stack->head,      // Expected value / Old value
      (size_t)next_node         // New value
    );

  } while(casRes != (size_t)stack->head);

#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  return 0;
}

