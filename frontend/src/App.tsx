import { BrowserRouter, Route, Routes } from "react-router-dom";
import { useState } from "react";
import Layout from "./components/Layout";
import LandingPage from "./pages/LandingPage";
import LiveInterpreterPage from "./pages/LiveInterpreterPage";
import ResultsPage from "./pages/ResultsPage";
import AnalyticsPage from "./pages/AnalyticsPage";
import AboutPage from "./pages/AboutPage";

export default function App() {
  const [highContrast, setHighContrast] = useState(false);

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout highContrast={highContrast} onToggleContrast={() => setHighContrast((prev) => !prev)} />}>
          <Route index element={<LandingPage />} />
          <Route path="interpreter" element={<LiveInterpreterPage />} />
          <Route path="results" element={<ResultsPage />} />
          <Route path="analytics" element={<AnalyticsPage />} />
          <Route path="about" element={<AboutPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
