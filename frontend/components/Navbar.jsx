import React from 'react';

const Navbar = () => {
  return (
    <nav style={styles.nav}>
      <div style={styles.inner}>
        <span style={styles.brand}>Movie Recommender</span>
        {/* Add your links / buttons here */}
      </div>
    </nav>
  );
};

const styles = {
  nav: {
    position: 'sticky', // stays in normal flow, sticks at top
    top: 0,
    zIndex: 1000,
    width: '100%',
    backdropFilter: 'blur(6px)',
    background: 'rgba(20,20,20,0.85)',
    borderBottom: '1px solid rgba(255,255,255,0.1)',
  },
  inner: {
    maxWidth: '1200px',
    margin: '0 auto',
    padding: '0.75rem 1.25rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    color: '#fff',
    fontFamily: 'system-ui, sans-serif',
  },
  brand: {
    fontSize: '1.1rem',
    fontWeight: 600,
    letterSpacing: '.5px',
  },
};

export default Navbar;
