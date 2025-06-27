import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface PanelState {
  isOpen: boolean;
  size: number; // percentage
  minSize: number;
  maxSize: number;
}

interface UIState {
  panels: {
    left: PanelState;
    center: PanelState;
    right: PanelState;
  };
  activeView: 'dashboard' | 'analytics' | 'settings' | 'export';
  isFullscreen: boolean;
  isSidebarCollapsed: boolean;
  theme: 'dark' | 'light';
  showPerformanceMonitor: boolean;
  expandedSections: Record<string, boolean>;
  modalStack: string[]; // Track open modals
  notifications: {
    enabled: boolean;
    soundEnabled: boolean;
  };
}

const initialState: UIState = {
  panels: {
    left: {
      isOpen: true,
      size: 25,
      minSize: 20,
      maxSize: 40,
    },
    center: {
      isOpen: true,
      size: 50,
      minSize: 30,
      maxSize: 70,
    },
    right: {
      isOpen: true,
      size: 25,
      minSize: 20,
      maxSize: 40,
    },
  },
  activeView: 'dashboard',
  isFullscreen: false,
  isSidebarCollapsed: false,
  theme: 'dark',
  showPerformanceMonitor: false,
  expandedSections: {
    agentTemplates: true,
    activeAgents: true,
    knowledgeGraph: true,
    analytics: false,
  },
  modalStack: [],
  notifications: {
    enabled: true,
    soundEnabled: false,
  },
};

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    // Panel management
    togglePanel: (state, action: PayloadAction<'left' | 'center' | 'right'>) => {
      const panel = state.panels[action.payload];
      panel.isOpen = !panel.isOpen;
    },

    resizePanel: (state, action: PayloadAction<{
      panel: 'left' | 'center' | 'right';
      size: number;
    }>) => {
      const { panel, size } = action.payload;
      const panelState = state.panels[panel];
      panelState.size = Math.max(panelState.minSize, Math.min(panelState.maxSize, size));
    },

    // View management
    setActiveView: (state, action: PayloadAction<UIState['activeView']>) => {
      state.activeView = action.payload;
    },

    toggleFullscreen: (state) => {
      state.isFullscreen = !state.isFullscreen;
    },

    toggleSidebar: (state) => {
      state.isSidebarCollapsed = !state.isSidebarCollapsed;
    },

    // Theme
    setTheme: (state, action: PayloadAction<'dark' | 'light'>) => {
      state.theme = action.payload;
    },

    // Performance monitor
    togglePerformanceMonitor: (state) => {
      state.showPerformanceMonitor = !state.showPerformanceMonitor;
    },

    // Section expansion
    toggleSection: (state, action: PayloadAction<string>) => {
      const section = action.payload;
      state.expandedSections[section] = !state.expandedSections[section];
    },

    setExpandedSections: (state, action: PayloadAction<Record<string, boolean>>) => {
      state.expandedSections = action.payload;
    },

    // Modal management
    openModal: (state, action: PayloadAction<string>) => {
      if (!state.modalStack.includes(action.payload)) {
        state.modalStack.push(action.payload);
      }
    },

    closeModal: (state, action: PayloadAction<string>) => {
      state.modalStack = state.modalStack.filter(id => id !== action.payload);
    },

    closeAllModals: (state) => {
      state.modalStack = [];
    },

    // Notifications
    toggleNotifications: (state) => {
      state.notifications.enabled = !state.notifications.enabled;
    },

    toggleSoundNotifications: (state) => {
      state.notifications.soundEnabled = !state.notifications.soundEnabled;
    },

    // Batch updates for layout
    updateLayout: (state, action: PayloadAction<{
      panels?: Partial<UIState['panels']>;
      activeView?: UIState['activeView'];
    }>) => {
      if (action.payload.panels) {
        Object.entries(action.payload.panels).forEach(([key, value]) => {
          if (state.panels[key as keyof UIState['panels']] && value) {
            state.panels[key as keyof UIState['panels']] = {
              ...state.panels[key as keyof UIState['panels']],
              ...value,
            };
          }
        });
      }
      if (action.payload.activeView) {
        state.activeView = action.payload.activeView;
      }
    },

    // Reset to default layout
    resetLayout: (state) => {
      state.panels = initialState.panels;
      state.isSidebarCollapsed = false;
      state.isFullscreen = false;
    },
  },
});

export const {
  togglePanel,
  resizePanel,
  setActiveView,
  toggleFullscreen,
  toggleSidebar,
  setTheme,
  togglePerformanceMonitor,
  toggleSection,
  setExpandedSections,
  openModal,
  closeModal,
  closeAllModals,
  toggleNotifications,
  toggleSoundNotifications,
  updateLayout,
  resetLayout,
} = uiSlice.actions;

export default uiSlice.reducer; 