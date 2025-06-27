import { configureStore } from '@reduxjs/toolkit';
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';
import agentReducer from './slices/agentSlice';
import conversationReducer from './slices/conversationSlice';
import uiReducer from './slices/uiSlice';
import connectionReducer from './slices/connectionSlice';
import knowledgeReducer from './slices/knowledgeSlice';
import analyticsReducer from './slices/analyticsSlice';

export const store = configureStore({
  reducer: {
    agents: agentReducer,
    conversations: conversationReducer,
    ui: uiReducer,
    connection: connectionReducer,
    knowledge: knowledgeReducer,
    analytics: analyticsReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types
        ignoredActions: ['socket/connected', 'socket/disconnected'],
        // Ignore these paths in the state
        ignoredPaths: ['connection.socket'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Typed hooks
export const useAppDispatch: () => AppDispatch = useDispatch;
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector; 