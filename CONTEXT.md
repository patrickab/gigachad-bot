# GigaChad Bot

A self-hosted LLM workspace for chats, documents, canvases, coding agents and learning.

## Language

### What the assistant knows

**Memory**:
User-confirmed facts about the user or a project, such as preferences, profile and project decisions.
_Avoid_: knowledge base, notes

**Knowledge**:
Objective propositions about the world, linked by reasoning chains, built up while learning. Global, shared by every project.
_Avoid_: memory, knowledge base

### Conversations

**Session**:
One conversation thread run by one engine. Every session is a chat session, code session or learn session.
_Avoid_: thread, conversation

**Chat session**:
A session with a local model through gigachad's own chat engine.
_Avoid_: chat (as the umbrella term)

**Code session**:
A session with an OMP coding agent.
_Avoid_: OMP session, omp tab

**Learn session**:
A session run by LearnHarness that measures, plans and teaches toward a learning goal.
_Avoid_: lesson, tutoring chat

**Learn mode**:
LearnHarness's way of working with a model, with all its functionality: learn sessions, practice sets and Knowledge.
_Avoid_: study mode, tutor mode
